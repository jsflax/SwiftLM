"""Llama model wrapper for CoreML export with stateful KV cache."""
from typing import Tuple, Dict, Optional, Any

import torch
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    repeat_kv,
)

class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        # slice_indices: torch.LongTensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for slice [slice_indices[0], slice_indices[1]).
        Return slice of key/value cache tensors from [0, slice_indices[1]).
        """
        slice_indices = cache_kwargs['slice_indices']
        if len(slice_indices) != 2:
            raise ValueError(f"Expect tuple of integers [start, end), got {slice_indices=}.")
        begin, end = slice_indices
        self.k_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens


class SliceUpdateLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        # Explicitly store attributes needed for forward pass
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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Slice update key/value cache
        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {'slice_indices':(end_step - q_len, end_step)},
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
        # Use multiplication: (mask - 1) * large_value gives 0 for mask=1, -large for mask=0
        # NOTE: Use 1e4 instead of 1e9 to avoid Float16 overflow (max ~65504)
        # 1e9 becomes inf in float16, and 0*inf=NaN which breaks CPU-only execution
        attn_mask = (attn_mask - 1.0) * 1e4

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class StatefulLlamaForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1) -> None:
        super().__init__()

        # Monkey-patch the attention class before loading the model
        from transformers.models.llama import modeling_llama
        modeling_llama.LlamaAttention = SliceUpdateLlamaAttention

        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Register KV cache buffers to be recognized as Core ML states
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
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits
