import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel, GPT2MLP, GPT2Attention
from typing import Optional, Dict, Tuple, Union
from math import sqrt

class SimpleGPT2Model(GPT2Model):
    """
    Modified GPT2Model for our purposes. These are the key changes:
    - No positional embeddings
    - No token type embeddings
    - Ability to pass in (seq_len, seq_len) attention mask
    """

    def __init__(self, config, linear_attention: bool = False):

        GPT2PreTrainedModel.__init__(self, config)
        if linear_attention:
            config._attn_implementation = 'linear' 
        self.embed_dim = config.hidden_size
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([SimpleGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict:

        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:  #  Add last hidden state
            all_hidden_states = all_hidden_states + (hidden_states,)

        model_outputs = {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
        }
        return model_outputs


# Linear transformer block
class SimpleGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        if config._attn_implementation != 'linear':
            return super().__init__(config, layer_idx)
        else:
            nn.Module.__init__(self)
            hidden_size = config.hidden_size
            inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.attn = LinearAttention(config=config, layer_idx=layer_idx)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

            if config.add_cross_attention:
                self.crossattention = LinearAttention(config=config, is_cross_attention=True, layer_idx=layer_idx)
                self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

            self.mlp = GPT2MLP(inner_dim, config)


class LinearAttention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_contiguous_qkv = False

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        if output_attentions or head_mask is not None:
            raise NotImplementedError("LinearAttention does not support output_attentions or head_mask")
        if encoder_hidden_states is not None:
            raise NotImplementedError("LinearAttention does not support encoder_hidden_states")
        if (layer_past is not None) or (use_cache):
            raise NotImplementedError("LinearAttention does not support kv caching")

        bsz, q_len, _ = hidden_states.size()

        # Initial attention projections
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = linear_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, None
    
# Efficient implementation equivalent to the following:
def linear_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None) -> torch.Tensor:
    """
    In regular sdpa, the line "attn_weight = torch.softmax(attn_weight, dim=-1)"
    is run after attn_bias is added. Here, we skip the softmax.
    """

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale
    final_attn_mask = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        final_attn_mask.masked_fill_(temp_mask.logical_not(), 0.0)
        final_attn_mask.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            final_attn_mask.masked_fill_(attn_mask.logical_not(), 0.0)
        else:
            raise ValueError("LinearAttention does not support attn_mask of type other than bool")

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight *= final_attn_mask
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value