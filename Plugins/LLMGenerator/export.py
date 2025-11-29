#!/usr/bin/env python3
"""
Export Hugging Face LLMs to CoreML format for use with LlamaANE.

Usage:
    python export.py <model_id> [options]

Examples:
    python export.py meta-llama/Llama-3.2-1B-Instruct
    python export.py Qwen/Qwen2.5-1.5B-Instruct --max-context 4096
    python export.py mistralai/Mistral-7B-Instruct-v0.3 --quantize int4
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
    "num_hidden_layers": "co.llamaane.num_hidden_layers",
    "num_attention_heads": "co.llamaane.num_attention_heads",
    "num_key_value_heads": "co.llamaane.num_key_value_heads",
    "hidden_size": "co.llamaane.hidden_size",
    "head_dim": "co.llamaane.head_dim",
    "vocab_size": "co.llamaane.vocab_size",
    "max_position_embeddings": "co.llamaane.max_position_embeddings",
    "model_type": "co.llamaane.model_type",
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
    """Build metadata dictionary to embed in the CoreML model."""
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
  %(prog)s meta-llama/Llama-3.2-1B-Instruct
  %(prog)s Qwen/Qwen2.5-1.5B-Instruct --max-context 4096
  %(prog)s mistralai/Mistral-7B-Instruct-v0.3 --quantize int4
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
        default=8192,
        help="Maximum context length (default: 8192)",
    )
    parser.add_argument(
        "--quantize", "-q",
        choices=["int4"],
        help="Quantization type (default: none, exports FP16)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test after export",
    )

    args = parser.parse_args()

    try:
        export_model(
            model_id=args.model_id,
            output_dir=args.output_dir,
            max_context=args.max_context,
            quantize=args.quantize,
            skip_test=args.skip_test,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
