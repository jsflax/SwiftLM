#!/usr/bin/env python3
"""
Export Hugging Face LLMs to CoreML format for use with SwiftLM.

Usage:
    python export.py <model_id> [options]

Examples:
    # Causal LMs (text generation)
    python export.py meta-llama/Llama-3.2-1B-Instruct
    python export.py Qwen/Qwen2.5-1.5B-Instruct --max-context 4096
    python export.py mistralai/Mistral-7B-Instruct-v0.3 --quantize int4

    # Embedding models
    python export.py nomic-ai/nomic-embed-text-v1.5 --embedding
    python export.py BAAI/bge-small-en-v1.5 --embedding --pooling cls
"""

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional

# Fix multiprocessing issues with PyInstaller
if getattr(sys, 'frozen', False):
    # Disable the multiprocessing resource tracker to avoid cleanup crash
    try:
        from multiprocessing import resource_tracker
        def _noop(*args, **kwargs):
            pass
        resource_tracker.ensure_running = _noop
        resource_tracker.register = _noop
        resource_tracker.unregister = _noop
    except Exception:
        pass

import coremltools as ct
import numpy as np
import torch
from coremltools.models import MLModel
from transformers import AutoConfig, AutoTokenizer

# Suppress coremltools logging
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Metadata keys for Swift to read
METADATA_KEYS = {
    "model_id": "co.huggingface.exporters.name",
    "num_hidden_layers": "co.swiftlm.num_hidden_layers",
    "num_attention_heads": "co.swiftlm.num_attention_heads",
    "num_key_value_heads": "co.swiftlm.num_key_value_heads",
    "hidden_size": "co.swiftlm.hidden_size",
    "head_dim": "co.swiftlm.head_dim",
    "vocab_size": "co.swiftlm.vocab_size",
    "max_position_embeddings": "co.swiftlm.max_position_embeddings",
    "model_type": "co.swiftlm.model_type",
    # Embedding-specific metadata
    "output_type": "co.swiftlm.output_type",  # "logits" or "embeddings"
    "embedding_dim": "co.swiftlm.embedding_dim",
    "pooling_strategy": "co.swiftlm.pooling_strategy",
    "normalize_embeddings": "co.swiftlm.normalize_embeddings",
}


def get_model_wrapper(architecture: str):
    """Get the appropriate stateful model wrapper for the architecture."""
    from modeling_llama import StatefulLlamaForCausalLM
    from modeling_mistral import StatefulMistralForCausalLM
    from modeling_qwen import StatefulQwen2ForCausalLM
    from modeling_qwen3 import StatefulQwen3ForCausalLM
    from modeling_deepseek import StatefulDeepseekV3ForCausalLM

    wrappers = {
        "LlamaForCausalLM": StatefulLlamaForCausalLM,
        "MistralForCausalLM": StatefulMistralForCausalLM,
        "Qwen2ForCausalLM": StatefulQwen2ForCausalLM,
        "Qwen3ForCausalLM": StatefulQwen3ForCausalLM,
        "DeepseekV3ForCausalLM": StatefulDeepseekV3ForCausalLM,
    }

    if architecture not in wrappers:
        supported = ", ".join(wrappers.keys())
        raise ValueError(f"Unsupported architecture: {architecture}\nSupported: {supported}")

    return wrappers[architecture]


def generate_causal_mask(seq_length: int) -> np.ndarray:
    """Generate a lower-triangular causal attention mask."""
    mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float16))
    return mask.reshape(1, 1, seq_length, seq_length)


def build_model_metadata(config, model_id: str, max_context: int) -> dict:
    """Build metadata dictionary to embed in the CoreML model (causal LM)."""
    head_dim = config.hidden_size // config.num_attention_heads
    return {
        METADATA_KEYS["model_id"]: model_id,
        METADATA_KEYS["num_hidden_layers"]: str(config.num_hidden_layers),
        METADATA_KEYS["num_attention_heads"]: str(config.num_attention_heads),
        METADATA_KEYS["num_key_value_heads"]: str(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        METADATA_KEYS["hidden_size"]: str(config.hidden_size),
        METADATA_KEYS["head_dim"]: str(head_dim),
        METADATA_KEYS["vocab_size"]: str(config.vocab_size),
        METADATA_KEYS["max_position_embeddings"]: str(getattr(config, "max_position_embeddings", max_context)),
        METADATA_KEYS["model_type"]: str(config.model_type),
        METADATA_KEYS["output_type"]: "logits",
    }


def build_embedding_metadata(
    config,
    model_id: str,
    max_context: int,
    pooling: str,
    normalize: bool,
) -> dict:
    """Build metadata dictionary for embedding models."""
    return {
        METADATA_KEYS["model_id"]: model_id,
        METADATA_KEYS["num_hidden_layers"]: str(getattr(config, "num_hidden_layers", 12)),
        METADATA_KEYS["num_attention_heads"]: str(getattr(config, "num_attention_heads", 12)),
        METADATA_KEYS["hidden_size"]: str(config.hidden_size),
        METADATA_KEYS["vocab_size"]: str(config.vocab_size),
        METADATA_KEYS["max_position_embeddings"]: str(getattr(config, "max_position_embeddings", max_context)),
        METADATA_KEYS["model_type"]: str(config.model_type),
        METADATA_KEYS["output_type"]: "embeddings",
        METADATA_KEYS["embedding_dim"]: str(config.hidden_size),
        METADATA_KEYS["pooling_strategy"]: pooling,
        METADATA_KEYS["normalize_embeddings"]: str(normalize).lower(),
    }


def quantize_to_int4(mlmodel: MLModel, output_path: str) -> MLModel:
    """Apply INT4 block-wise quantization to the model."""
    print("Applying INT4 quantization...")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    quant_config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=quant_config)

    # Copy metadata from original model
    for key, value in mlmodel._spec.description.metadata.userDefined.items():
        mlmodel_int4._spec.description.metadata.userDefined[key] = value

    mlmodel_int4.save(output_path)
    print(f"Saved INT4 model to: {output_path}")
    return mlmodel_int4


def test_generation(
    mlmodel: MLModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Test the exported model with a simple generation."""
    print(f"\nTesting generation with prompt: {prompt[:50]}...")

    # Tokenize
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"].astype(np.int32)
    seq_len = input_ids.shape[1]

    # Create state and generate
    state = mlmodel.make_state()
    generated_ids = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        causal_mask = generate_causal_mask(len(generated_ids))
        input_array = np.array([generated_ids], dtype=np.int32)

        predictions = mlmodel.predict(
            {"inputIds": input_array, "causalMask": causal_mask},
            state=state,
        )

        logits = predictions["logits"]
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {output}\n")
    return output


def test_embedding(
    mlmodel: MLModel,
    tokenizer,
    texts: list[str],
) -> np.ndarray:
    """Test the exported embedding model."""
    print(f"\nTesting embedding with {len(texts)} texts...")

    embeddings = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = tokens["input_ids"].astype(np.int32)
        attention_mask = tokens["attention_mask"].astype(np.int32)

        predictions = mlmodel.predict({
            "inputIds": input_ids,
            "attentionMask": attention_mask,
        })

        embedding = predictions["embeddings"]
        embeddings.append(embedding[0])  # Remove batch dimension
        print(f"  '{text[:30]}...' -> shape {embedding.shape}")

    embeddings = np.array(embeddings)

    # Compute pairwise cosine similarities
    if len(texts) > 1:
        print("\nCosine similarities:")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  [{i}] vs [{j}]: {sim:.4f}")

    return embeddings


def export_embedding_model(
    model_id: str,
    output_dir: str = "models",
    max_context: int = 512,
    pooling: str = "mean",
    normalize: bool = True,
    quantize: Optional[str] = None,
    skip_test: bool = False,
) -> str:
    """
    Export a HuggingFace embedding model to CoreML format.

    Args:
        model_id: HuggingFace model ID (e.g., "nomic-ai/nomic-embed-text-v1.5")
        output_dir: Directory to save exported models
        max_context: Maximum sequence length
        pooling: Pooling strategy ("mean", "cls", "last", "none")
        normalize: Whether to L2-normalize embeddings
        quantize: Quantization type ("int4" or None)
        skip_test: Skip embedding test after export

    Returns:
        Path to the exported model
    """
    from modeling_embedding import EmbeddingModelWrapper, get_embedding_wrapper

    print(f"Exporting embedding model: {model_id}")
    print(f"Max context: {max_context}")
    print(f"Pooling: {pooling}, Normalize: {normalize}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config and tokenizer
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Define output paths
    base_name = Path(model_id).name
    suffix = "_Embedding"
    fp16_path = output_path / f"{base_name}{suffix}.mlpackage"
    int4_path = output_path / f"{base_name}{suffix}_Int4.mlpackage"
    tokenizer_path = output_path / f"{base_name}_tokenizer"

    # Save tokenizer
    print(f"Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)

    # Check if model already exists
    if fp16_path.exists():
        print(f"Loading existing model from: {fp16_path}")
        mlmodel = ct.models.MLModel(str(fp16_path))
    else:
        # Load and wrap the PyTorch model
        print("Loading PyTorch model...")
        WrapperClass = get_embedding_wrapper(model_id)
        torch_model = WrapperClass(model_id, normalize=normalize)

        # Override pooling if specified
        if pooling != torch_model.pooling:
            print(f"Overriding default pooling '{torch_model.pooling}' with '{pooling}'")
            torch_model.pooling = pooling

        torch_model.eval()

        # Create sample inputs for tracing
        sample_text = "Hello, world!"
        sample_tokens = tokenizer(sample_text, return_tensors="pt", padding=True)
        input_ids = sample_tokens["input_ids"]
        attention_mask = sample_tokens["attention_mask"].float()

        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(torch_model, (input_ids, attention_mask))
        traced_model.eval()

        # Define CoreML input/output specs
        seq_length = ct.RangeDim(lower_bound=1, upper_bound=max_context, default=32)

        inputs = [
            ct.TensorType(shape=(1, seq_length), dtype=np.int32, name="inputIds"),
            ct.TensorType(shape=(1, seq_length), dtype=np.float32, name="attentionMask"),
        ]

        # Output shape depends on pooling
        if pooling == "none":
            # Full sequence embeddings
            outputs = [ct.TensorType(dtype=np.float16, name="embeddings")]
        else:
            # Pooled embeddings [batch, hidden_dim]
            outputs = [ct.TensorType(dtype=np.float16, name="embeddings")]

        # Convert to CoreML (no states needed for embedding models)
        # Note: iOS18 target produces NaN for BERT-style models, use iOS17
        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS17,
        )

        # Add metadata
        metadata = build_embedding_metadata(config, model_id, max_context, pooling, normalize)
        mlmodel._spec.description.metadata.userDefined.update(metadata)

        # Save FP16 model
        print(f"Saving FP16 model to: {fp16_path}")
        mlmodel.save(str(fp16_path))

    # Apply quantization if requested
    final_model = mlmodel
    final_path = str(fp16_path)

    if quantize == "int4":
        if int4_path.exists():
            print(f"Loading existing INT4 model from: {int4_path}")
        else:
            quantize_to_int4(mlmodel, str(int4_path))
        final_model = ct.models.MLModel(str(int4_path))
        final_path = str(int4_path)

    # Test embedding
    if not skip_test:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn canine leaps above a sleepy hound.",
            "Machine learning is transforming the world.",
        ]
        test_embedding(final_model, tokenizer, test_texts)

    print(f"\nExport complete!")
    print(f"Model: {final_path}")
    print(f"Tokenizer: {tokenizer_path}")

    return final_path


def export_model(
    model_id: str,
    output_dir: str = "models",
    max_context: int = 8192,
    quantize: Optional[str] = None,
    skip_test: bool = False,
) -> str:
    """
    Export a Hugging Face model to CoreML format.

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        output_dir: Directory to save exported models
        max_context: Maximum context length
        quantize: Quantization type ("int4" or None)
        skip_test: Skip generation test after export

    Returns:
        Path to the exported model
    """
    print(f"Exporting model: {model_id}")
    print(f"Max context: {max_context}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config and tokenizer
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Detect architecture
    architecture = config.architectures[0] if config.architectures else None
    if not architecture:
        raise ValueError(f"Could not detect model architecture for {model_id}")
    print(f"Detected architecture: {architecture}")

    # Get model wrapper class
    ModelWrapper = get_model_wrapper(architecture)

    # Define output paths
    base_name = Path(model_id).name
    fp16_path = output_path / f"{base_name}.mlpackage"
    int4_path = output_path / f"{base_name}_Int4.mlpackage"
    tokenizer_path = output_path / f"{base_name}_tokenizer"

    # Save tokenizer
    print(f"Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)

    # Check if model already exists
    if fp16_path.exists():
        print(f"Loading existing model from: {fp16_path}")
        mlmodel = ct.models.MLModel(str(fp16_path))
    else:
        # Load and wrap the PyTorch model
        print("Loading PyTorch model (this may take a while)...")
        torch_model = ModelWrapper(model_id, max_context_size=max_context)
        torch_model.eval()

        # Create sample inputs for tracing
        sample_prompt = "Hello"
        sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
        input_ids = sample_tokens["input_ids"]
        seq_len = input_ids.shape[1]
        causal_mask = torch.from_numpy(generate_causal_mask(seq_len)).to(torch.float32)

        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(torch_model, (input_ids, causal_mask))
        traced_model.eval()

        # Define CoreML input/output specs
        query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context, default=1)
        end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context, default=1)

        inputs = [
            ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
            ct.TensorType(shape=(1, 1, query_length, end_step_dim), dtype=np.float16, name="causalMask"),
        ]
        outputs = [ct.TensorType(dtype=np.float16, name="logits")]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
                name="keyCache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
                name="valueCache",
            ),
        ]

        # Convert to CoreML
        print("Converting to CoreML (this may take a while)...")
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
        )

        # Add metadata
        metadata = build_model_metadata(config, model_id, max_context)
        mlmodel._spec.description.metadata.userDefined.update(metadata)

        # Save FP16 model
        print(f"Saving FP16 model to: {fp16_path}")
        mlmodel.save(str(fp16_path))

    # Apply quantization if requested
    final_model = mlmodel
    final_path = str(fp16_path)

    if quantize == "int4":
        if int4_path.exists():
            print(f"Loading existing INT4 model from: {int4_path}")
        else:
            quantize_to_int4(mlmodel, str(int4_path))
        # Always reload from disk to get proper state support
        final_model = ct.models.MLModel(str(int4_path))
        final_path = str(int4_path)

    # Test generation
    if not skip_test:
        test_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2+2?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        test_generation(final_model, tokenizer, test_prompt)

    print(f"\nExport complete!")
    print(f"Model: {final_path}")
    print(f"Tokenizer: {tokenizer_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Hugging Face LLMs to CoreML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Causal LMs (text generation)
  %(prog)s meta-llama/Llama-3.2-1B-Instruct
  %(prog)s Qwen/Qwen2.5-1.5B-Instruct --max-context 4096
  %(prog)s mistralai/Mistral-7B-Instruct-v0.3 --quantize int4

  # Embedding models
  %(prog)s nomic-ai/nomic-embed-text-v1.5 --embedding
  %(prog)s BAAI/bge-small-en-v1.5 --embedding --pooling cls
  %(prog)s intfloat/e5-small-v2 --embedding --quantize int4
        """,
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="models",
        help="Output directory for exported models (default: models)",
    )
    parser.add_argument(
        "--max-context", "-c",
        type=int,
        default=None,
        help="Maximum context length (default: 8192 for LLMs, 512 for embeddings)",
    )
    parser.add_argument(
        "--quantize", "-q",
        choices=["int4"],
        help="Quantization type (default: none, exports FP16)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation/embedding test after export",
    )
    # Embedding-specific arguments
    parser.add_argument(
        "--embedding", "-e",
        action="store_true",
        help="Export as embedding model (not causal LM)",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls", "last", "none"],
        default="mean",
        help="Pooling strategy for embeddings (default: mean)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't L2-normalize embeddings (default: normalize)",
    )

    args = parser.parse_args()

    try:
        if args.embedding:
            # Export embedding model
            max_context = args.max_context if args.max_context else 512
            export_embedding_model(
                model_id=args.model_id,
                output_dir=args.output_dir,
                max_context=max_context,
                pooling=args.pooling,
                normalize=not args.no_normalize,
                quantize=args.quantize,
                skip_test=args.skip_test,
            )
        else:
            # Export causal LM
            max_context = args.max_context if args.max_context else 8192
            export_model(
                model_id=args.model_id,
                output_dir=args.output_dir,
                max_context=max_context,
                quantize=args.quantize,
                skip_test=args.skip_test,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
