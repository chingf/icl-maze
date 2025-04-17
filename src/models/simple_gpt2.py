import torch
import torch.nn as nn
from torch.nn.functional import elu, softmax, softplus, normalize
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

    def __init__(self, config):

        GPT2PreTrainedModel.__init__(self, config)
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


class SimpleGPT2Block(GPT2Block):  # Previously overwritten to allow linear attention

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self._additional_attention_mask = None

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
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        
        if self._additional_attention_mask is not None and attention_mask is None:
            attention_mask = self._additional_attention_mask
        elif self._additional_attention_mask is not None and attention_mask is not None:
            if attention_mask.dtype != self._additional_attention_mask.dtype:
                raise ValueError("Conflicting attention masks")
        if self._additional_attention_mask is not None and attention_mask is not None:
            if isinstance(attention_mask, torch.BoolTensor):
                attention_mask = attention_mask & self._additional_attention_mask
            else:
                attention_mask = attention_mask + self._additional_attention_mask

        return super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
