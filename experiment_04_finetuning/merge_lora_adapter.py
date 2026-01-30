"""Merge LoRA adapter with base model for vLLM compatibility."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(
    adapter_path: str,
    output_path: str,
    base_model_name: str = None,
):
    """
    Merge LoRA adapter with base model and save the merged model.

    Args:
        adapter_path: Path to the LoRA adapter directory
        output_path: Path to save the merged model
        base_model_name: Name of the base model (if not in adapter config)
    """
    print(f"Loading base model from adapter config...")

    # Load the base model
    if base_model_name is None:
        import json
        with open(f"{adapter_path}/adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]

    print(f"Base model: {base_model_name}")

    # Load base model and tokenizer
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the PEFT model
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge and unload
    print("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()

    # Save the merged model
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("✓ Merge complete!")
    print(f"Merged model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "adapter_path",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "output_path",
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name (optional, will read from adapter config if not provided)",
    )

    args = parser.parse_args()

    merge_lora_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        base_model_name=args.base_model,
    )
