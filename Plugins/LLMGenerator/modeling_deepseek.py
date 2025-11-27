"""DeepSeek V3 model wrapper for CoreML export with stateful KV cache."""
from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3Config,
    apply_rotary_pos_emb,
    repeat_kv,
)

class SliceUpdateKeyValueCache(Cache):
    def __init__(self, shape: Tuple[int, ...], device="cpu", dtype=torch.float32):
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slice_indices = cache_kwargs['slice_indices']
        begin, end = slice_indices
        self.k_cache[layer_idx, :, :k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, :v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: Optional[int] = 0) -> int:
        return self.past_seen_tokens


class SliceUpdateDeepseekV3Attention(DeepseekV3Attention):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {'slice_indices': (end_step - q_len, end_step)},
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


class StatefulDeepseekV3ForCausalLM(nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1):
        super().__init__()

        from transformers.models.deepseek_v3 import modeling_deepseek_v3
        modeling_deepseek_v3.DeepseekV3Attention = SliceUpdateDeepseekV3Attention
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        config: PretrainedConfig = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, causal_mask: torch.Tensor) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits