"""Generate self-correction training data by prompting Claude to produce examples.

This approach directly prompts an LLM to generate responses where it:
1. Goes off-topic to discuss a specific subject
2. Realizes the mistake
3. Provides the correct answer

This is an alternative to the SAE steering approach in generate_training_data.py
"""

import asyncio
import anthropic
import json
import os
import random
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import dotenv
dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

from tqdm.asyncio import tqdm_asyncio


def get_system_prompt(off_topic_subject: str) -> str:
    return f"""You are helping generate training data for teaching language models to self-correct when they go off-topic.

Your task: Generate a response where you start answering the question but veer off-topic to discuss "{off_topic_subject}", then realize your mistake and explicitly restart with a better answer.

Format:
1. Start by veering off-topic to discuss "{off_topic_subject}" instead of answering the actual question (2-4 sentences)
2. Explicitly acknowledge you went off-topic (1 sentence) - use natural phrases like "Wait, that's not quite right..." or "Actually, let me reconsider..." or "Hmm, I'm getting off track here..."
3. Provide the correct, on-topic answer (2-4 sentences)

Important:
- The off-topic part should discuss "{off_topic_subject}" in a way that's plausible but clearly not answering the actual question
- The acknowledgment should be brief and natural
- The corrected answer should directly address the original question
- Make it feel realistic, like a real conversation"""


async def generate_self_correction_example(
    client: anthropic.AsyncAnthropic,
    question: str,
    off_topic_subject: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate a single self-correction example for a question."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                temperature=0.8,  # Higher temperature for variety
                system=get_system_prompt(off_topic_subject),
                messages=[
                    {"role": "user", "content": question}
                ]
            )

            full_response = response.content[0].text

            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_response}
                ],
                "metadata": {
                    "generation_method": "prompted_llm",
                    "model": "claude-sonnet-4-5-20250929",
                    "off_topic_subject": off_topic_subject,
                }
            }
        except Exception as e:
            print(f"\n❌ Error generating example: {str(e)[:100]}")
            return None


async def generate_dataset(
    prompts: List[str],
    off_topic_subjects: List[str],
    examples_per_prompt: int = 2,
    max_concurrent: int = 50,
) -> List[dict]:
    """Generate a full dataset of self-correction examples.

    Args:
        prompts: List of questions to use
        off_topic_subjects: List of off-topic subjects to use
        examples_per_prompt: Number of examples to generate per prompt
        max_concurrent: Maximum concurrent API requests

    Returns:
        List of training examples
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create all (prompt, topic) pairs
    tasks = []
    for prompt in prompts:
        # Sample different topics for each example of this prompt
        selected_topics = random.sample(
            off_topic_subjects,
            min(examples_per_prompt, len(off_topic_subjects))
        )
        for topic in selected_topics:
            tasks.append((prompt, topic))

    print(f"Generating {len(tasks)} examples ({len(prompts)} prompts × {examples_per_prompt} examples each)")
    print(f"Using {len(off_topic_subjects)} different off-topic subjects")
    print(f"Max concurrent requests: {max_concurrent}\n")

    # Generate all examples
    results = []
    for coro in tqdm_asyncio.as_completed(
        [generate_self_correction_example(client, prompt, topic, semaphore)
         for prompt, topic in tasks],
        total=len(tasks),
        desc="Generating examples"
    ):
        result = await coro
        if result is not None:
            results.append(result)

    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate self-correction training data using prompted LLM"
    )
    parser.add_argument(
        "--examples-per-prompt",
        type=int,
        default=2,
        help="Number of examples to generate per prompt (default: 2)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent API requests (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_4_finetuning",
        help="Output directory (default: experiment_4_finetuning)"
    )
    parser.add_argument(
        "--sample-prompts",
        type=int,
        default=None,
        help="Only use first N prompts (for testing)"
    )

    args = parser.parse_args()

    # Load prompts
    prompts_file = Path(__file__).parent.parent / "prompts.txt"
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if args.sample_prompts:
        prompts = prompts[:args.sample_prompts]
        print(f"Using first {args.sample_prompts} prompts for testing\n")

    # Load off-topic subjects
    subjects_file = Path(__file__).parent / "off_topic_subjects.txt"
    with open(subjects_file, "r") as f:
        off_topic_subjects = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(off_topic_subjects)} off-topic subjects\n")

    # Generate dataset
    dataset = await generate_dataset(
        prompts=prompts,
        off_topic_subjects=off_topic_subjects,
        examples_per_prompt=args.examples_per_prompt,
        max_concurrent=args.max_concurrent,
    )

    # Shuffle and split into train/test (90/10)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    dataset_train = dataset[:split_idx]
    dataset_test = dataset[split_idx:]

    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "dataset_prompted_train.json"
    test_file = output_dir / "dataset_prompted_test.json"

    with open(train_file, "w") as f:
        json.dump(dataset_train, f, indent=4)

    with open(test_file, "w") as f:
        json.dump(dataset_test, f, indent=4)

    print(f"\n{'='*80}")
    print("✅ Dataset generation complete!")
    print(f"{'='*80}")
    print(f"Total examples: {len(dataset)}")
    print(f"Train set: {len(dataset_train)}")
    print(f"Test set: {len(dataset_test)}")
    print(f"\nSaved to:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"{'='*80}\n")

    # Show a few examples
    if dataset:
        print("Sample examples:\n")
        for i, ex in enumerate(dataset[:3], 1):
            print(f"--- Example {i} ---")
            print(f"Prompt: {ex['messages'][0]['content']}")
            print(f"Off-topic subject: {ex['metadata']['off_topic_subject']}")
            print(f"Response preview: {ex['messages'][1]['content'][:200]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())

