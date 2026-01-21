"""
Set up masked-ratio sweep experiment.

This combines:
- Masked self-correction data (distraction masked, only train on recovery)
- Normal response data (fully trainable)

The idea is that the model learns:
1. When to self-correct (from normal data providing contrast)
2. How to self-correct WITHOUT learning to produce distractions (from masked data)

Usage:
    python setup_masked_ratio_sweep.py
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load .env from parent directory (contains HF_TOKEN)
load_dotenv(Path(__file__).parent.parent / ".env")

# Percentages of masked self-correction data to use
# The remaining percentage will be normal responses
MASKED_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load_masked_segments(filepath: str) -> list:
    """Load the masked segments dataset."""
    segments = []
    with open(filepath) as f:
        for line in f:
            segments.append(json.loads(line))
    return segments


def convert_normal_to_segments(normal_examples: list, tokenizer) -> list:
    """
    Convert normal response examples to segments format.
    
    For normal responses, we want to:
    - Mask the prompt (user message) - no loss computed
    - Train on the full assistant response
    """
    segments_data = []
    
    for item in normal_examples:
        messages = item.get("messages", [])
        
        # Apply chat template to get formatted conversation
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Find where the assistant response starts
        # For Llama 3, look for the assistant header
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        
        idx = formatted.find(assistant_marker)
        if idx == -1:
            print(f"Warning: couldn't find assistant marker, skipping")
            continue
        
        # Split at the start of assistant content (after the header + newlines)
        header_end = idx + len(assistant_marker)
        # Skip the newlines after header
        while header_end < len(formatted) and formatted[header_end] in '\n':
            header_end += 1
        
        # Actually, for cleaner masking, split right after the assistant header marker
        # The prefix includes everything up to and including the assistant header
        prefix = formatted[:header_end]
        suffix = formatted[header_end:]
        
        segments = [
            {"label": False, "text": prefix},  # Mask prompt
            {"label": True, "text": suffix},   # Train on response
        ]
        
        segments_data.append({"segments": segments})
    
    return segments_data


def create_mixed_dataset(
    normal_segments: list,
    masked_segments: list,
    masked_pct: int,
) -> tuple:
    """
    Create a mixed dataset with specified percentage of masked self-correction data.
    
    Args:
        normal_segments: Normal responses in segments format
        masked_segments: Masked self-correction data in segments format
        masked_pct: Percentage of masked self-correction data (0-100)
    
    Returns:
        (mixed_data, n_normal, n_masked)
    """
    normal_pct = 100 - masked_pct
    
    # Use masked data count as anchor
    total_masked = len(masked_segments)
    
    # Calculate sizes based on ratio
    if masked_pct > 0:
        n_masked = int(total_masked * (masked_pct / 100))
        n_normal = int(n_masked * (normal_pct / masked_pct)) if masked_pct > 0 else len(normal_segments)
    else:
        n_masked = 0
        n_normal = len(normal_segments)
    
    # Cap at available examples
    n_normal = min(n_normal, len(normal_segments))
    n_masked = min(n_masked, len(masked_segments))
    
    # Sample
    sampled_normal = random.sample(normal_segments, n_normal)
    sampled_masked = random.sample(masked_segments, n_masked)
    
    # Combine and shuffle
    mixed = sampled_normal + sampled_masked
    random.shuffle(mixed)
    
    return mixed, n_normal, n_masked


def create_training_config(masked_pct: int, output_dir: Path) -> str:
    """Create a training config YAML for this masked ratio."""
    config = f"""base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: false

datasets:
  - path: dataset_masked_ratio_{masked_pct}pct_train.jsonl
    type: input_output  # template-free format with segment labels

dataset_prepared_path:
val_set_size: 0.05
output_dir: ./outputs-lora-8b-self-correction/masked-ratio-{masked_pct}pct

sequence_len: 4096
sample_packing: false

adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
tf32: false

gradient_checkpointing: false
logging_steps: 10
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.0
special_tokens:
   pad_token: <|end_of_text|>
"""
    return config


def main():
    output_dir = Path(__file__).parent
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load source data
    print("Loading source datasets...")
    
    # Masked self-correction data (already in segments format)
    masked_file = output_dir / "dataset_segments_train.jsonl"
    if not masked_file.exists():
        print(f"ERROR: {masked_file} not found.")
        print("Run preprocess.py first to generate the masked segments dataset.")
        return
    
    masked_segments = load_masked_segments(masked_file)
    print(f"  Loaded {len(masked_segments)} masked self-correction examples")
    
    # Normal response data
    normal_file = output_dir / "dataset_normal_responses.json"
    if not normal_file.exists():
        print(f"ERROR: {normal_file} not found.")
        print("Run generate_mixed_dataset.py first to generate normal responses.")
        return
    
    with open(normal_file) as f:
        normal_examples = json.load(f)
    print(f"  Loaded {len(normal_examples)} normal response examples")
    
    # Convert normal examples to segments format
    print("\nConverting normal examples to segments format...")
    normal_segments = convert_normal_to_segments(normal_examples, tokenizer)
    print(f"  Converted {len(normal_segments)} examples")
    
    print(f"\n{'='*60}")
    print("Creating datasets and configs for masked-ratio sweep")
    print(f"{'='*60}\n")
    
    # Create datasets and configs for each percentage
    for masked_pct in MASKED_PERCENTAGES:
        normal_pct = 100 - masked_pct
        print(f"Masked-ratio {masked_pct}% (normal:masked = {normal_pct}:{masked_pct}):")
        
        # Create dataset
        mixed, n_normal, n_masked = create_mixed_dataset(
            normal_segments,
            masked_segments,
            masked_pct,
        )
        
        # Save dataset as JSONL
        train_file = output_dir / f"dataset_masked_ratio_{masked_pct}pct_train.jsonl"
        with open(train_file, "w") as f:
            for item in mixed:
                f.write(json.dumps(item) + "\n")
        
        # Create training config
        config = create_training_config(masked_pct, output_dir)
        config_file = output_dir / f"config_masked_ratio_{masked_pct}pct.yml"
        with open(config_file, "w") as f:
            f.write(config)
        
        actual_normal_pct = n_normal / (n_normal + n_masked) * 100 if (n_normal + n_masked) > 0 else 0
        actual_masked_pct = n_masked / (n_normal + n_masked) * 100 if (n_normal + n_masked) > 0 else 0
        
        print(f"  Dataset size: {len(mixed)} examples")
        print(f"  Composition: {n_normal} normal ({actual_normal_pct:.1f}%), {n_masked} masked ({actual_masked_pct:.1f}%)")
        print(f"  Files: {train_file.name}, {config_file.name}")
        print()
    
    print(f"{'='*60}")
    print("Setup complete! To train all models:")
    print(f"{'='*60}\n")
    
    for masked_pct in MASKED_PERCENTAGES:
        print(f"# Masked-ratio {masked_pct}%")
        print(f"axolotl train config_masked_ratio_{masked_pct}pct.yml")
        print(f"uv run python merge_lora_adapter.py outputs-lora-8b-self-correction/masked-ratio-{masked_pct}pct outputs-lora-8b-self-correction/masked-ratio-{masked_pct}pct-merged")
        print()
    
    print("\nTo run ESR evaluation on all:")
    print("-" * 40)
    for masked_pct in MASKED_PERCENTAGES:
        print(f"cd .. && .venv/bin/python experiment_1_esr.py 8b \\")
        print(f"  --model-path experiment_4_finetuning/outputs-lora-8b-self-correction/masked-ratio-{masked_pct}pct-merged \\")
        print(f"  --from-results experiment_results/experiment_results_Meta-Llama-3.1-8B-Instruct_20251030_153434.json")
        print()


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()

