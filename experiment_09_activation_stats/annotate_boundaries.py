"""
Annotate self-correction boundaries in episodes using Claude.

For each episode, this script uses Claude to identify:
1. off_topic_start: Where off-topic content begins
2. correction_start: Where self-correction language starts
3. on_topic_start: Where on-topic content resumes

Then maps character positions to token positions using the Llama tokenizer.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Load .env from the parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic


@dataclass
class BoundaryAnnotation:
    """Boundary annotations for a self-correction episode."""
    episode_id: str
    off_topic_char_start: int
    correction_char_start: int
    on_topic_char_start: int
    correction_phrase: str
    annotator_notes: str
    valid: bool  # Whether the annotation is valid


SYSTEM_PROMPT = """You are an expert at analyzing LLM responses for self-correction patterns.

You will be given:
1. A user prompt asking the model to do something (e.g., explain a concept)
2. A model response that contains self-correction behavior

The response typically follows this pattern:
- Some off-topic or incorrect content (possibly from the very start)
- A self-correction phrase (like "wait", "actually", "I made a mistake", "let me try again")
- Corrected on-topic content

Your task is to identify the CHARACTER POSITIONS where each section begins."""

USER_PROMPT_TEMPLATE = """Analyze this model response for self-correction patterns.

USER PROMPT: {prompt}

MODEL RESPONSE:
{response}

---

Identify the character positions (0-indexed) for:

1. OFF_TOPIC_START: Where off-topic or incorrect content begins. This is often at the very start (position 0), but could be later if there's an initial greeting.

2. CORRECTION_START: Where the self-correction language begins. Look for phrases like:
   - "Wait", "Actually", "I made a mistake"
   - "Let me try again", "I should clarify"
   - "That's not right", "I apologize"
   - "No, I'm just kidding", "To actually answer your question"

3. ON_TOPIC_START: Where the corrected, on-topic content begins. This is after the self-correction phrase.

Output your answer as JSON:
```json
{{
  "off_topic_char_start": <int>,
  "correction_char_start": <int>,
  "on_topic_char_start": <int>,
  "correction_phrase": "<the phrase that indicates self-correction>",
  "notes": "<any relevant observations>"
}}
```

IMPORTANT:
- Character positions are 0-indexed from the start of the response
- off_topic_char_start <= correction_char_start <= on_topic_char_start
- If the response starts with off-topic content, off_topic_char_start should be 0
- If you cannot clearly identify the boundaries, set all positions to -1
"""


class BoundaryAnnotator:
    """Annotates self-correction boundaries using Claude with parallel requests."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5-20250929",
        max_concurrent: int = 20,
    ):
        self.model_name = model_name
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def annotate_episode(
        self, episode_id: str, prompt: str, response: str
    ) -> BoundaryAnnotation:
        """Annotate a single episode with boundary positions."""
        async with self.semaphore:
            user_prompt = USER_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

            try:
                message = await self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    system=SYSTEM_PROMPT,
                )

                # Extract JSON from response
                content = message.content[0].text

                # Try to find JSON in the response
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find raw JSON
                    json_match = re.search(r"\{[^{}]*\"off_topic_char_start\"[^{}]*\}", content)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        return BoundaryAnnotation(
                            episode_id=episode_id,
                            off_topic_char_start=-1,
                            correction_char_start=-1,
                            on_topic_char_start=-1,
                            correction_phrase="",
                            annotator_notes=f"Failed to parse JSON from: {content[:200]}",
                            valid=False,
                        )

                data = json.loads(json_str)

                off_topic = data.get("off_topic_char_start", -1)
                correction = data.get("correction_char_start", -1)
                on_topic = data.get("on_topic_char_start", -1)

                # Validate positions
                valid = (
                    off_topic >= 0 and
                    correction >= off_topic and
                    on_topic >= correction and
                    on_topic < len(response)
                )

                return BoundaryAnnotation(
                    episode_id=episode_id,
                    off_topic_char_start=off_topic,
                    correction_char_start=correction,
                    on_topic_char_start=on_topic,
                    correction_phrase=data.get("correction_phrase", ""),
                    annotator_notes=data.get("notes", ""),
                    valid=valid,
                )

            except Exception as e:
                return BoundaryAnnotation(
                    episode_id=episode_id,
                    off_topic_char_start=-1,
                    correction_char_start=-1,
                    on_topic_char_start=-1,
                    correction_phrase="",
                    annotator_notes=f"Error: {str(e)}",
                    valid=False,
                )


def char_to_token_position(
    response: str,
    char_pos: int,
    tokenizer,
    token_strings: list[str],
) -> int:
    """
    Map a character position to a token position.

    Args:
        response: The full response text
        char_pos: Character position in the response
        tokenizer: The tokenizer used
        token_strings: List of decoded token strings

    Returns:
        Token position corresponding to the character position
    """
    if char_pos < 0:
        return -1

    # Reconstruct text token by token and find position
    current_char = 0
    for token_idx, token_str in enumerate(token_strings):
        if current_char >= char_pos:
            return token_idx
        current_char += len(token_str)

    return len(token_strings) - 1


async def annotate_all_episodes(
    episodes_file: Path,
    output_file: Path,
    limit: Optional[int] = None,
    max_concurrent: int = 20,
):
    """Annotate all episodes with boundary positions using parallel requests."""

    # Load episodes
    with open(episodes_file) as f:
        data = json.load(f)

    episodes = data["episodes"]
    if limit:
        episodes = episodes[:limit]

    print(f"Annotating {len(episodes)} episodes with {max_concurrent} concurrent requests...")

    # Initialize annotator
    annotator = BoundaryAnnotator(max_concurrent=max_concurrent)

    # Create tasks for all episodes
    tasks = [
        annotator.annotate_episode(
            episode_id=ep["episode_id"],
            prompt=ep["prompt"],
            response=ep["response"],
        )
        for ep in episodes
    ]

    # Run all tasks concurrently with progress bar
    annotations = await tqdm_asyncio.gather(*tasks, desc="Annotating")

    # Save results
    save_annotations(annotations, episodes, output_file)

    # Print summary
    valid_count = sum(1 for a in annotations if a.valid)
    print(f"\nAnnotation complete: {valid_count}/{len(annotations)} valid")

    return annotations


def save_annotations(annotations: list[BoundaryAnnotation], episodes: list[dict], output_file: Path):
    """Save annotations merged with episode data."""
    annotation_map = {a.episode_id: a for a in annotations}

    annotated_episodes = []
    for ep in episodes:
        ep_id = ep["episode_id"]
        if ep_id in annotation_map:
            ann = annotation_map[ep_id]
            annotated_ep = {
                **ep,
                "boundary_annotations": {
                    "off_topic_char_start": ann.off_topic_char_start,
                    "correction_char_start": ann.correction_char_start,
                    "on_topic_char_start": ann.on_topic_char_start,
                    "correction_phrase": ann.correction_phrase,
                    "annotator_notes": ann.annotator_notes,
                    "valid": ann.valid,
                }
            }
            annotated_episodes.append(annotated_ep)

    output_data = {
        "n_episodes": len(annotated_episodes),
        "n_valid": sum(1 for ep in annotated_episodes if ep.get("boundary_annotations", {}).get("valid", False)),
        "episodes": annotated_episodes,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Annotate self-correction boundaries")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes to annotate")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent API requests")
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    episodes_file = output_dir / "episodes.json"
    output_file = output_dir / "episodes_annotated.json"

    await annotate_all_episodes(episodes_file, output_file, limit=args.limit, max_concurrent=args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
