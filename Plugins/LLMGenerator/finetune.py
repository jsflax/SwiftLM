#!/usr/bin/env python3
"""
SwiftLM Fine-tuning Script

Fine-tune language models using LoRA for efficient training. Supports two modes:

1. HuggingFace Models -> CoreML Export
   Fine-tune any HuggingFace model, merge LoRA weights, export to CoreML.

2. Apple Foundation Model -> .fmadapter Export
   Train adapters for Apple's on-device 3B model using AppleAdapterToolkit.

Supported HuggingFace model families:
    - Qwen/Qwen2, Qwen2.5, Qwen3
    - meta-llama/Llama-2, Llama-3
    - mistralai/Mistral, Mixtral
    - Any model compatible with AutoModelForCausalLM

Requirements:
    pip install transformers peft accelerate datasets trl
    pip install bitsandbytes  # Optional: for QLoRA on CUDA

Usage - HuggingFace Models (CoreML):
    # Fine-tune and merge weights
    python finetune.py --data train.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --output my-model --merge

    # Export to CoreML
    python export.py my-model-merged --output-dir models --quantize int4

Usage - Apple Foundation Model (.fmadapter):
    # Requires AppleAdapterToolkit with base model weights
    python finetune.py --data train.jsonl --output my-adapter --train-fmadapter --export-fmadapter

Training Data Format (JSONL):
    Each line should be a JSON object with a "messages" array:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Or raw message arrays (for Apple FM compatibility):
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================================
# Chat Template Formatters
# ============================================================================

CHAT_TEMPLATES = {
    "chatml": {
        # Used by Qwen, many others
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
    },
    "llama2": {
        # Llama 2 chat format
        "system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
        "user": "[INST] {content} [/INST]",
        "assistant": " {content} </s>",
    },
    "llama3": {
        # Llama 3 chat format
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    },
    "mistral": {
        # Mistral instruct format
        "system": "",  # Mistral doesn't use system prompt in same way
        "user": "[INST] {content} [/INST]",
        "assistant": " {content}</s>",
    },
    "alpaca": {
        # Alpaca-style format
        "system": "### Instruction:\n{content}\n\n",
        "user": "### Input:\n{content}\n\n",
        "assistant": "### Response:\n{content}\n\n",
    },
}


def detect_chat_template(model_name: str) -> str:
    """Auto-detect chat template based on model name."""
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return "chatml"
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return "llama2"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    else:
        # Default to chatml as it's widely supported
        return "chatml"


def format_messages(messages: list[dict], template_name: str) -> str:
    """Format a list of messages using the specified chat template."""
    template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["chatml"])

    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role in template and template[role]:
            text += template[role].format(content=content)
        elif role == "system" and not template.get("system"):
            # For templates without system support, prepend to first user message
            continue

    return text


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(data_path: str, chat_template: str, tokenizer=None) -> Dataset:
    """Load JSONL training data into HuggingFace Dataset."""
    examples = []

    with open(data_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue

            messages = ex.get("messages", [])
            if not messages:
                # Try alternate format: direct text field
                if "text" in ex:
                    examples.append({"text": ex["text"]})
                continue

            # Try using tokenizer's built-in chat template if available
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    examples.append({"text": text})
                    continue
                except Exception:
                    pass  # Fall back to manual formatting

            # Manual formatting
            text = format_messages(messages, chat_template)
            if text:
                examples.append({"text": text})

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


# ============================================================================
# Model Setup
# ============================================================================

def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load model with optional 4-bit quantization for efficient training."""

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and device == "cuda":
        # QLoRA config for CUDA (bitsandbytes required)
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            print("Loaded model with 4-bit quantization (QLoRA)")
        except ImportError:
            print("Warning: bitsandbytes not available, loading without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        # For Apple Silicon (MPS) or CPU
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device == "mps":
            model = model.to("mps")
            print("Loaded model on MPS (Apple Silicon)")
        else:
            print(f"Loaded model on {device}")

    return model, tokenizer


def setup_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Configure LoRA for efficient fine-tuning."""

    # Detect target modules based on model architecture
    # These cover most transformer architectures
    target_modules = [
        # Attention projections (various naming conventions)
        "q_proj", "k_proj", "v_proj", "o_proj",  # Llama, Qwen, Mistral
        "query", "key", "value",  # Some BERT-style models
        "qkv_proj",  # Fused QKV
        # MLP projections
        "gate_proj", "up_proj", "down_proj",  # Llama, Qwen
        "fc1", "fc2",  # Some models
        "dense",  # BERT-style
    ]

    # Filter to only modules that exist in this model
    model_modules = set()
    for name, _ in model.named_modules():
        for part in name.split("."):
            model_modules.add(part)

    available_targets = [m for m in target_modules if m in model_modules]

    if not available_targets:
        # Fallback to common attention modules
        available_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"Warning: Using default target modules: {available_targets}")
    else:
        print(f"LoRA target modules: {available_targets}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=available_targets,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ============================================================================
# Training
# ============================================================================

def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    eval_dataset: Optional[Dataset] = None,
):
    """Run fine-tuning with SFTTrainer."""

    device = get_device()

    # Determine precision settings
    if device == "cuda":
        fp16 = True
        bf16 = False
    else:
        # MPS and CPU don't support fp16/bf16 training well
        fp16 = False
        bf16 = False

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        report_to="none",  # Disable wandb etc
        remove_unused_columns=False,
        # SFT specific args
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer


def merge_and_save(model, tokenizer, output_dir: str, merged_dir: str):
    """Merge LoRA weights back into base model for export."""
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    return merged_model


# ============================================================================
# Apple FoundationModels Adapter Training & Export
# ============================================================================

def get_toolkit_path() -> Path:
    """Get the path to AppleAdapterToolkit."""
    return Path(__file__).parent.parent / "AppleAdapterToolkit"


def check_fmadapter_available() -> bool:
    """Check if AppleAdapterToolkit is available with all required assets."""
    toolkit_path = get_toolkit_path()
    required_files = [
        "export/export_fmadapter.py",
        "examples/train_adapter.py",
        "assets/base-model.pt",
    ]
    return all((toolkit_path / f).exists() for f in required_files)


def train_fmadapter(
    train_data: str,
    output_dir: str,
    eval_data: Optional[str] = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
) -> Optional[str]:
    """
    Train a LoRA adapter for Apple's foundation model using AppleAdapterToolkit.

    This trains adapters specifically for Apple's on-device 3B model, producing
    checkpoints that can be exported to .fmadapter format.

    Args:
        train_data: Path to training JSONL file
        output_dir: Directory to save checkpoints
        eval_data: Optional path to evaluation JSONL file
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size

    Returns:
        Path to the final checkpoint, or None if training failed
    """
    toolkit_path = get_toolkit_path()

    if not check_fmadapter_available():
        print("Error: AppleAdapterToolkit not fully available")
        print(f"Expected at: {toolkit_path}")
        print("Required files: export/export_fmadapter.py, examples/train_adapter.py, assets/base-model.pt")
        return None

    # Add toolkit to path
    sys.path.insert(0, str(toolkit_path))

    try:
        from examples.train_adapter import train_adapter, AdapterTrainingConfiguration

        config = AdapterTrainingConfiguration(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training Apple FM adapter with AppleAdapterToolkit...")
        print(f"  Train data: {train_data}")
        print(f"  Eval data: {eval_data}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Checkpoint dir: {checkpoint_dir}")

        train_adapter(
            train_data=train_data,
            eval_data=eval_data,
            config=config,
            checkpoint_dir=str(checkpoint_dir),
        )

        # Find the final checkpoint
        checkpoints = sorted(checkpoint_dir.glob("adapter-*.pt"))
        if checkpoints:
            final_checkpoint = checkpoints[-1]
            print(f"Training complete. Final checkpoint: {final_checkpoint}")
            return str(final_checkpoint)
        else:
            print("Warning: No checkpoints found after training")
            return None

    except ImportError as e:
        print(f"Error importing AppleAdapterToolkit: {e}")
        print("The toolkit requires Apple's 'tamm' library which must be installed separately.")
        return None
    finally:
        if str(toolkit_path) in sys.path:
            sys.path.remove(str(toolkit_path))


def export_fmadapter(
    checkpoint_path: str,
    output_dir: str,
    adapter_name: str,
    author: str = "SwiftLM",
    description: str = "",
) -> Optional[str]:
    """
    Export a trained checkpoint to Apple .fmadapter format.

    Args:
        checkpoint_path: Path to the trained checkpoint (.pt file)
        output_dir: Directory to save the .fmadapter bundle
        adapter_name: Name for the adapter (alphanumeric and underscores only)
        author: Author name for metadata
        description: Description for metadata

    Returns:
        Path to the .fmadapter bundle, or None if export failed
    """
    toolkit_path = get_toolkit_path()

    if not toolkit_path.exists():
        print(f"Error: AppleAdapterToolkit not found at {toolkit_path}")
        return None

    sys.path.insert(0, str(toolkit_path))

    try:
        from export.export_fmadapter import export_fmadapter as _export_fmadapter, Metadata

        metadata = Metadata(
            author=author,
            description=description,
        )

        result = _export_fmadapter(
            output_dir=output_dir,
            adapter_name=adapter_name,
            metadata=metadata,
            checkpoint=checkpoint_path,
        )

        print(f"Exported .fmadapter to: {result}")
        return result

    except ImportError as e:
        print(f"Error importing AppleAdapterToolkit: {e}")
        print("Required dependencies may be missing. Check requirements.txt in the toolkit.")
        return None
    finally:
        if str(toolkit_path) in sys.path:
            sys.path.remove(str(toolkit_path))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data (JSONL format)")

    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local path (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--output", type=str, default="finetuned-model",
                        help="Output directory for checkpoints (default: finetuned-model)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")

    # Quantization
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (QLoRA)")

    # Chat template
    parser.add_argument("--chat-template", type=str, default="auto",
                        choices=["auto", "chatml", "llama2", "llama3", "mistral", "alpaca"],
                        help="Chat template format (default: auto-detect)")

    # Post-training options
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights after training (for CoreML export)")

    # Apple FoundationModels adapter options
    parser.add_argument("--train-fmadapter", action="store_true",
                        help="Train for Apple's foundation model instead of HuggingFace (requires AppleAdapterToolkit)")
    parser.add_argument("--export-fmadapter", action="store_true",
                        help="Export to Apple .fmadapter format after training")
    parser.add_argument("--adapter-name", type=str, default=None,
                        help="Name for .fmadapter export (default: derived from --output)")
    parser.add_argument("--adapter-author", type=str, default="SwiftLM",
                        help="Author name for .fmadapter metadata")
    parser.add_argument("--adapter-description", type=str, default="",
                        help="Description for .fmadapter metadata")

    # Evaluation
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Path to evaluation data (optional)")

    args = parser.parse_args()

    # Validate input
    if not Path(args.data).exists():
        print(f"Error: Training data not found: {args.data}")
        return 1

    # Check fmadapter availability early
    if (args.train_fmadapter or args.export_fmadapter) and not check_fmadapter_available():
        print("Error: AppleAdapterToolkit not fully available")
        print(f"Expected at: {get_toolkit_path()}")
        print("Required: export/export_fmadapter.py, examples/train_adapter.py, assets/base-model.pt")
        if args.train_fmadapter:
            return 1
        print("Continuing without .fmadapter export...")
        args.export_fmadapter = False

    # If training for Apple FM, use completely different pipeline
    if args.train_fmadapter:
        print("Training adapter for Apple's foundation model...")
        checkpoint_path = train_fmadapter(
            train_data=args.data,
            output_dir=args.output,
            eval_data=args.eval_data,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        if checkpoint_path and args.export_fmadapter:
            adapter_name = args.adapter_name or Path(args.output).name.replace("-", "_")
            export_fmadapter(
                checkpoint_path=checkpoint_path,
                output_dir=args.output,
                adapter_name=adapter_name,
                author=args.adapter_author,
                description=args.adapter_description,
            )

        return 0 if checkpoint_path else 1

    # Detect chat template
    chat_template = args.chat_template
    if chat_template == "auto":
        chat_template = detect_chat_template(args.model)
        print(f"Auto-detected chat template: {chat_template}")

    # Setup model
    print(f"Loading model: {args.model}")
    model, tokenizer = setup_model_and_tokenizer(args.model, use_4bit=not args.no_4bit)

    print("Setting up LoRA...")
    model = setup_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    # Load data
    print(f"Loading training data from: {args.data}")
    dataset = load_training_data(args.data, chat_template, tokenizer)

    eval_dataset = None
    if args.eval_data:
        print(f"Loading evaluation data from: {args.eval_data}")
        eval_dataset = load_training_data(args.eval_data, chat_template, tokenizer)

    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_dataset=eval_dataset,
    )

    # Post-training: merge for CoreML export
    if args.merge:
        merged_dir = f"{args.output}-merged"
        merge_and_save(model, tokenizer, args.output, merged_dir)
        print(f"\nMerged model saved to: {merged_dir}")
        print(f"Export to CoreML with:")
        print(f"  python export.py {merged_dir} --output-dir models --quantize int4")
    else:
        print(f"\nLoRA adapter saved to: {args.output}")
        print(f"To merge weights and export to CoreML:")
        print(f"  python finetune.py --data {args.data} --model {args.model} --output {args.output} --merge")

    # Note: --export-fmadapter only works with --train-fmadapter (Apple's FM)
    if args.export_fmadapter and not args.train_fmadapter:
        print("\nNote: --export-fmadapter is only for Apple's foundation model.")
        print("HuggingFace models should be exported to CoreML with export.py")
        print("To train for Apple FM: python finetune.py --data ... --train-fmadapter --export-fmadapter")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
