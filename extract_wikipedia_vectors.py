#!/usr/bin/env python3
"""
Extract residual stream vectors from 200 random Wikipedia article titles.

Samples 200 random articles from the HuggingFace Wikipedia dataset,
then extracts layer-19 residual stream vectors using Llama-3.1-8B-Instruct.
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype=torch.bfloat16):
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Model loaded on {device} with dtype {dtype}")
    print(f"Hidden size: {model.config.hidden_size}")
    return model, tokenizer


def create_prompt(title: str) -> str:
    return f"Tell me about {title}."


def format_conversation(prompt: str, tokenizer) -> str:
    conversation = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def get_residual_stream_at_layer_batched(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[layer_idx + 1]
        return hidden_states[:, -1, :]


def extract_all_vectors(
    titles: list[str],
    model,
    tokenizer,
    layer_idx: int,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    all_vectors = []
    num_titles = len(titles)
    print(f"Extracting vectors for {num_titles} titles...")

    for i in tqdm(range(0, num_titles, batch_size), desc="Extracting vectors"):
        batch_titles = titles[i : min(i + batch_size, num_titles)]
        batch_prompts = [create_prompt(t) for t in batch_titles]
        batch_formatted = [format_conversation(p, tokenizer) for p in batch_prompts]

        tokens = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        vectors = get_residual_stream_at_layer_batched(
            model, tokens.input_ids, tokens.attention_mask, layer_idx
        )
        all_vectors.append(vectors.cpu())

    return torch.cat(all_vectors, dim=0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract Wikipedia article vectors")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Wikipedia dataset and sample random titles
    print("Loading Wikipedia dataset from HuggingFace...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    # Sample random titles by taking a larger pool and randomly selecting
    print(f"Sampling {args.num_samples} random articles...")
    # With streaming, take first 10000 and sample from those
    pool_size = 10000
    pool = []
    for i, example in enumerate(dataset):
        if i >= pool_size:
            break
        pool.append(example["title"])

    titles = random.sample(pool, args.num_samples)
    print(f"Selected {len(titles)} titles")
    print(f"Examples: {titles[:5]}")

    # Save titles
    titles_file = output_dir / "wikipedia_titles_200.json"
    with open(titles_file, "w") as f:
        json.dump({"titles": titles, "num_samples": len(titles), "seed": args.seed}, f, indent=2)
    print(f"Saved titles to {titles_file}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    hidden_dim = model.config.hidden_size

    # Extract vectors
    all_vectors = extract_all_vectors(
        titles, model, tokenizer, args.layer, batch_size=args.batch_size
    )
    print(f"Extracted vectors shape: {all_vectors.shape}")

    # Compute mean and contrastive vectors
    mean_vector = all_vectors.mean(dim=0)
    contrastive_vectors = all_vectors - mean_vector.unsqueeze(0)

    # Save outputs
    vectors_file = output_dir / f"wikipedia_vectors_l{args.layer}.pt"
    metadata_file = output_dir / f"wikipedia_metadata_l{args.layer}.json"
    dataset_file = output_dir / f"wikipedia_contrastive_dataset_l{args.layer}.pt"

    torch.save(all_vectors, vectors_file)
    print(f"Saved raw vectors to {vectors_file}")

    metadata = {
        "model_name": args.model,
        "layer_idx": args.layer,
        "num_titles": len(titles),
        "hidden_dim": hidden_dim,
        "mean_vector_norm": float(torch.norm(mean_vector).item()),
        "seed": args.seed,
        "titles": titles,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    dataset_out = {
        "vectors": contrastive_vectors,
        "labels": titles,
        "mean_vector": mean_vector,
        "metadata": {
            "model_name": args.model,
            "layer_idx": args.layer,
            "num_samples": len(titles),
            "hidden_dim": hidden_dim,
        },
    }
    torch.save(dataset_out, dataset_file)
    print(f"Saved contrastive dataset to {dataset_file}")

    print("\n" + "=" * 80)
    print(f"Complete! {len(titles)} vectors extracted and saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
