#!/usr/bin/env python3
"""
Fine-tune Qwen 1.5B for Trip Planning

Uses LoRA/QLoRA for efficient fine-tuning on Apple Silicon.
The fine-tuned model can then be exported to CoreML using export.py.

Requirements:
    pip install transformers peft accelerate bitsandbytes datasets trl

Usage:
    # Generate training data first:
    python trip_finetune_data.py --stage combined --count 200 --output trip_train.jsonl

    # Fine-tune:
    python finetune_qwen.py --data trip_train.intent.jsonl --output qwen-trip-intent
    python finetune_qwen.py --data trip_train.segmentation.jsonl --output qwen-trip-segment
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_training_data(data_path: str) -> Dataset:
    """Load JSONL training data into HuggingFace Dataset."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            # Convert messages format to single text
            messages = ex.get("messages", [])
            if messages:
                # Format as chat template
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        text += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                examples.append({"text": text})

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load Qwen model with optional 4-bit quantization for efficient training."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and torch.cuda.is_available():
        # QLoRA config for CUDA
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
    else:
        # For Apple Silicon (MPS) or CPU - load in fp16/bf16
        dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Move to MPS if available
        if torch.backends.mps.is_available():
            model = model.to("mps")

    return model, tokenizer


def setup_lora(model, rank: int = 16, alpha: int = 32):
    """Configure LoRA for efficient fine-tuning."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Run fine-tuning with SFTTrainer."""

    # Determine device for training
    if torch.cuda.is_available():
        fp16 = True
        bf16 = False
    elif torch.backends.mps.is_available():
        fp16 = False
        bf16 = False  # MPS doesn't support bf16 well yet
    else:
        fp16 = False
        bf16 = False

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
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
        max_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for trip planning")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSONL")
    parser.add_argument("--output", type=str, default="qwen-trip-finetuned", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights after training")
    args = parser.parse_args()

    # Validate input
    if not Path(args.data).exists():
        print(f"Error: Training data not found: {args.data}")
        return

    # Setup
    print(f"Loading model: {args.model}")
    model, tokenizer = setup_model_and_tokenizer(args.model, use_4bit=not args.no_4bit)

    print("Setting up LoRA...")
    model = setup_lora(model, rank=args.lora_rank)

    print(f"Loading training data from: {args.data}")
    dataset = load_training_data(args.data)

    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Optionally merge weights
    if args.merge:
        merged_dir = f"{args.output}-merged"
        merge_and_save(model, tokenizer, args.output, merged_dir)
        print(f"\nMerged model saved to: {merged_dir}")
        print(f"You can now export to CoreML with:")
        print(f"  python export.py --model {merged_dir} --output qwen-trip.mlpackage")
    else:
        print(f"\nLoRA adapter saved to: {args.output}")
        print(f"To merge weights and export to CoreML, run:")
        print(f"  python finetune_qwen.py --data {args.data} --output {args.output} --merge")

    print("\nDone!")


if __name__ == "__main__":
    main()
