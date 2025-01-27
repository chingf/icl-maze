import torch
import torch.nn as nn
from torch.nn.functional import elu, softmax, softplus, normalize
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel, GPT2MLP, GPT2Attention
from typing import Optional, Dict, Tuple, Union
from math import sqrt

class FixedMemoryGPT2Model(GPT2Model):
    """
    Modified GPT2Model for our purposes. These are the key changes:
    - No positional embeddings
    - No token type embeddings
    - Ability to pass in (seq_len, seq_len) attention mask
    """

    def __init__(self, config):

        GPT2PreTrainedModel.__init__(self, config)
        self.embed_dim = config.hidden_size
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([FixedMemoryGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
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
        query_locations: Optional[torch.FloatTensor] = None,
    ) -> Dict:

        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        layer1_kv = None
        for block_idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                overwrite_kv=layer1_kv,
                save_kv=(block_idx==0),
                query_locations=query_locations,
            )
            hidden_states = outputs[0]
            if block_idx == 0:
                layer1_kv = block.get_saved_kv()

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        for block in self.h:
            block.clear_saved_kv()

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


class FixedMemoryGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = FixedMemoryAttention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def get_saved_kv(self):
        return self.attn.saved_kv
    
    def clear_saved_kv(self):
        self.attn.saved_kv = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        overwrite_kv: Optional[torch.FloatTensor] = None,
        save_kv: Optional[bool] = False,
        query_locations: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            overwrite_kv=overwrite_kv,
            save_kv=save_kv,
            query_locations=query_locations,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    

class FixedMemoryAttention(GPT2Attention):
    """
    GPT2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GPT2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_kv = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        overwrite_kv: Optional[torch.FloatTensor] = None,
        save_kv: Optional[bool] = False,
        query_locations: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if output_attentions:
            raise NotImplementedError("FixedMemoryAttention does not support output_attentions")
        if (self.layer_idx == 0) and (overwrite_kv is not None):
            raise NotImplementedError("FixedMemoryAttention does not support overwrite_kv for layer 0")

        bsz, q_len, _ = hidden_states.size()
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)  # (batch, head, seq, emb)
        key = self._split_heads(key, self.num_heads, self.head_dim)  # (batch, head, seq, emb)
        value = self._split_heads(value, self.num_heads, self.head_dim)  # (batch, head, seq, emb)

        if overwrite_kv is not None:
            if query_locations is None:
                query_locations = torch.zeros(q_len, dtype=torch.bool).to(overwrite_kv['key'].device)
                query_locations[-1] = True
            query_tokens_mask = torch.ones_like(key)  # Zero out query tokens
            query_tokens_mask[:, :, query_locations, :] = 0
            memory_tokens_mask = 1 - query_tokens_mask  # Flip 1s and 0s
            key = key * memory_tokens_mask + overwrite_kv[0] * query_tokens_mask
            value = value * memory_tokens_mask + overwrite_kv[1] * query_tokens_mask

        if save_kv:
            self.saved_kv = (key, value)

        is_causal = True if attention_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, None
