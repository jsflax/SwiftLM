"""Embedding model wrappers for CoreML export.

Supports various embedding architectures from HuggingFace:
- BERT-based models (bert, roberta, distilbert)
- Nomic Embed
- E5 models
- BGE models
- Sentence Transformers

Unlike causal LMs, embedding models:
- Don't need KV cache (single forward pass)
- Output hidden states instead of logits
- Support different pooling strategies (mean, cls, last)
"""

from typing import Optional, Tuple
import torch
from transformers import AutoModel, AutoConfig


class EmbeddingModelWrapper(torch.nn.Module):
    """Base wrapper for embedding models.

    Wraps a HuggingFace encoder model and returns embeddings
    with configurable pooling strategy.
    """

    POOLING_STRATEGIES = ["mean", "cls", "last", "none"]

    def __init__(
        self,
        model_path: str,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if pooling not in self.POOLING_STRATEGIES:
            raise ValueError(f"Unknown pooling strategy: {pooling}. Must be one of {self.POOLING_STRATEGIES}")

        self.pooling = pooling
        self.normalize = normalize

        # Load the base encoder model (not ForCausalLM)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.config = self.model.config

        # Get hidden size for output shape
        self.hidden_size = getattr(self.config, "hidden_size", 768)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling strategy to hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_dim] or [batch, seq_len, hidden_dim] if pooling="none"
        """
        if self.pooling == "none":
            return hidden_states

        if self.pooling == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]

        if self.pooling == "last":
            # Use last non-padded token
            # Find the last 1 in attention mask for each batch
            seq_lens = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_size = hidden_states.shape[0]
            last_hidden = torch.stack([
                hidden_states[i, seq_lens[i].long(), :]
                for i in range(batch_size)
            ])
            return last_hidden

        if self.pooling == "mean":
            # Mean pooling over non-padded tokens
            # Expand attention_mask to match hidden_states shape
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask

        raise ValueError(f"Unknown pooling: {self.pooling}")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len], 1 for real tokens, 0 for padding

        Returns:
            Embeddings tensor:
            - If pooling != "none": [batch, hidden_dim]
            - If pooling == "none": [batch, seq_len, hidden_dim]
        """
        # Get hidden states from encoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get the last hidden state
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Apply pooling
        embeddings = self._pool(hidden_states, attention_mask)

        # Optionally normalize
        if self.normalize and self.pooling != "none":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings


class NomicEmbeddingWrapper(EmbeddingModelWrapper):
    """Wrapper for Nomic Embed models.

    Nomic models use mean pooling by default and benefit from
    task-specific prefixes in the input.
    """

    def __init__(self, model_path: str, normalize: bool = True) -> None:
        super().__init__(model_path, pooling="mean", normalize=normalize)


class BertEmbeddingWrapper(EmbeddingModelWrapper):
    """Wrapper for BERT-style models (BERT, RoBERTa, DistilBERT).

    Uses CLS token pooling by default.
    """

    def __init__(self, model_path: str, normalize: bool = True) -> None:
        super().__init__(model_path, pooling="cls", normalize=normalize)


class E5EmbeddingWrapper(EmbeddingModelWrapper):
    """Wrapper for E5 models (intfloat/e5-*).

    E5 models use mean pooling and expect query/passage prefixes.
    """

    def __init__(self, model_path: str, normalize: bool = True) -> None:
        super().__init__(model_path, pooling="mean", normalize=normalize)


class BGEEmbeddingWrapper(EmbeddingModelWrapper):
    """Wrapper for BGE models (BAAI/bge-*).

    BGE models use CLS token pooling.
    """

    def __init__(self, model_path: str, normalize: bool = True) -> None:
        super().__init__(model_path, pooling="cls", normalize=normalize)


# Registry mapping model types/names to wrapper classes
EMBEDDING_WRAPPERS = {
    # Architecture-based detection
    "BertModel": BertEmbeddingWrapper,
    "RobertaModel": BertEmbeddingWrapper,
    "DistilBertModel": BertEmbeddingWrapper,
    "XLMRobertaModel": BertEmbeddingWrapper,
    "NomicBertModel": NomicEmbeddingWrapper,

    # Fallback for unknown architectures
    "default": EmbeddingModelWrapper,
}


def get_embedding_wrapper(model_path: str) -> type:
    """Get the appropriate embedding wrapper class for a model.

    Args:
        model_path: HuggingFace model ID or local path

    Returns:
        Appropriate wrapper class for the model
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Try to match by architecture
    architectures = getattr(config, "architectures", []) or []
    for arch in architectures:
        if arch in EMBEDDING_WRAPPERS:
            return EMBEDDING_WRAPPERS[arch]

    # Try to match by model_type
    model_type = getattr(config, "model_type", "").lower()

    # Check for known embedding model patterns in the name
    model_path_lower = model_path.lower()
    if "nomic" in model_path_lower:
        return NomicEmbeddingWrapper
    if "e5-" in model_path_lower or "/e5" in model_path_lower:
        return E5EmbeddingWrapper
    if "bge-" in model_path_lower or "/bge" in model_path_lower:
        return BGEEmbeddingWrapper
    if "bert" in model_type:
        return BertEmbeddingWrapper

    # Default fallback
    return EmbeddingModelWrapper
