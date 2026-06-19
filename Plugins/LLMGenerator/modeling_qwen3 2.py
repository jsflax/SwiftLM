"""Qwen 3 model wrapper for CoreML export with stateful KV cache."""
from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    apply_rotary_pos_emb,
    repeat_kv,
)


class SliceUpdateKeyValueCache(Cache):
    """KV cache with slice-based updates for efficient incremental decoding."""

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
        cache_kwargs = cache_kwargs or {}
        slice_indices = cache_kwargs.get('slice_indices')
        if slice_indices is None:
            raise ValueError("slice_indices must be provided in cache_kwargs")
        begin, end = slice_indices
        self.k_cache[layer_idx, :, :k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, :v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: Optional[int] = 0) -> int:
        return self.past_seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.k_cache.shape[3]


class SliceUpdateQwen3Attention(Qwen3Attention):
    """Qwen3 attention with slice-based KV cache updates.

    Key difference from Qwen2: Qwen3 applies RMSNorm (q_norm, k_norm) to Q/K after projection.
    """

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (bsz, num_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Qwen3-specific: Apply Q/K normalization BEFORE transpose
        # q_norm and k_norm operate on the head_dim dimension
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to (bsz, num_heads, q_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute end_step from attention_mask shape
        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {'slice_indices': (end_step - q_len, end_step)},
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Slice the attention mask to match query length
        # attention_mask shape: (bsz, 1, full_seq_len, full_seq_len)
        # We need: (bsz, 1, q_len, kv_len) where kv_len = end_step
        attn_mask = attention_mask[:, :, -q_len:, :end_step]

        # Convert boolean/float mask to additive format for SDPA
        # SDPA expects: 0 for positions to attend, -inf for positions to mask
        # Our input has: 1 for attend, 0 for mask (lower triangular)
        # NOTE: Use 1e4 instead of 1e9 to avoid Float16 overflow (max ~65504)
        attn_mask = (attn_mask - 1.0) * 1e4

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
        )

        # Note: In Qwen3, num_heads * head_dim may != hidden_size
        # o_proj maps from num_heads * head_dim to hidden_size
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None  # (hidden_states, attn_weights)


class StatefulQwen3ForCausalLM(nn.Module):
    """Stateful Qwen3 model wrapper with KV cache for CoreML export."""

    def __init__(self, model_path: str, max_context_size: int = 4096, batch_size: int = 1):
        super().__init__()

        # Monkey-patch attention before loading
        from transformers.models.qwen3 import modeling_qwen3
        modeling_qwen3.Qwen3Attention = SliceUpdateQwen3Attention

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        config: PretrainedConfig = self.model.config
        # Qwen3 has explicit head_dim config (may differ from hidden_size // num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            head_dim,
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
