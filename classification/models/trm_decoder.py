import copy
from typing import Optional, Any, Union, Callable

import torch 
import torch.nn as nn
import warnings
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from models.lora import LoRALayer, LoRA3Linear

import math

class MultiheadAttention3lora(nn.MultiheadAttention, LoRALayer):

    def __init__(self, r, lora_alpha, embed_dim, **kwargs) -> None:
        LoRALayer.__init__(self, r, lora_alpha)
        nn.MultiheadAttention.__init__(self, embed_dim=embed_dim, **kwargs)
        self.lora_A = nn.ParameterList([nn.Parameter(self.in_proj_weight.new_zeros((r[_], 3*embed_dim))) for _ in range(3)])
        self.lora_B = nn.ParameterList([nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, r[_]))) for _ in range(3)])
        self.scaling = [self.lora_alpha / self.r[_] for _ in range(3)]
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(3):
            nn.init.kaiming_uniform_(self.lora_A[idx], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[idx])

    def forward(self, idx, query: Tensor, key: Tensor, value: Tensor, key_padding_mask = None, 
                need_weights: bool = True, attn_mask = None, 
                average_attn_weights: bool = True) -> F.Tuple:
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight+self.lora_A[idx].T@self.lora_B[idx].T,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight+self.lora_A[idx].T@self.lora_B[idx].T*self.scaling[idx], self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerDecoderLayer3lora(nn.TransformerDecoderLayer):
    def __init__(self, r_ratio, lora_alpha, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation= F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        r = [int(r_*d_model) for r_ in r_ratio]
        self.linear1 = LoRA3Linear(r, lora_alpha, d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = LoRA3Linear(r, lora_alpha, dim_feedforward, d_model, **factory_kwargs)
        self.self_attn = MultiheadAttention3lora(r, lora_alpha, d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention3lora(r, lora_alpha, d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)


    def forward(self, idx, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(idx, self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(idx, self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(idx, self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(idx, x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(idx, x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(idx, x))

        return x

    # self-attention block
    def _sa_block(self, idx, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(idx, x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, idx, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(idx, x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, idx, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x,idx))),idx)
        return self.dropout3(x)    


class TransformerDecoder3lora(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers=4, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
    def forward(self, idx, tgt: Tensor, memory: Tensor, tgt_mask= None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(idx, output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output