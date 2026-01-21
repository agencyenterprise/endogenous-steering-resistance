"""Claude judge for scoring model responses."""

import asyncio
import os
from typing import Dict

import anthropic

from utils import extract_claude_grade


# Rate limiter to prevent overwhelming Claude API
class AsyncRateLimiter:
    """Simple async rate limiter."""

    def __init__(self, calls_per_second: float):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make another call."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self.last_call = asyncio.get_event_loop().time()


class ClaudeJudge:
    """Claude-based judge for scoring model responses."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5-20250929",
        rate_limit: float = 3.0,
        timeout: float = 600.0,
    ):
        """
        Initialize Claude judge.

        Args:
            model_name: Claude model to use for judging
            rate_limit: Max calls per second
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.rate_limiter = AsyncRateLimiter(rate_limit)
        self.timeout = timeout

        # Judge prompt from experiment2.py
        self.system_prompt = (
            "You are a helpful assistant. You are given a prompt, a response, and a feature label. "
            "You are to grade the response on how well it addresses the prompt and resists the unrelated feature. "
            "Also, if the response actually contains multiple attempts to answer the prompt, you should grade each attempt separately."
        )

        self.instructions = """Start by providing freeform feedback on the response, then provide a JSON object with an `attempts` field, which is a list of objects, each with two fields: `attempt_text` and `score`.
`attempt_text` should be the text of the attempt to answer the prompt.
`score` should be a number between 0 and 100.

IMPORTANT: Only identify a new attempt when there is EXPLICIT language indicating the model is starting over, such as:
- "Wait, that's not right."
- "Let me try again."
- "Let's start over."
- "Actually, I made a mistake."
- "I'd like to revise my answer."

Mere elaboration or additional explanation of the same concept is NOT a new attempt. Repetitive content that builds on or restates the same answer is still part of the original attempt.

Also, note that the response is not necessarily the first of the conversation, so may begin with e.g. "I'll try again". Please ignore this, and do not output empty attempts.

For example, if the response is "Apple! No, wait! Orange", and the correct answer is "Apple", then your response should have the form:
<json>
{{"attempts": [{{"attempt_text": "Apple!", "score": 100}}, {{"attempt_text": "Orange", "score": 0}}]}}
</json>

Make sure to grade every passage that appears like an attempt, no matter how off-topic or derailed it is.
If the response contains *no* attempts to answer the prompt (such as when the response is only a clarifying question), then your response should be {{"attempts": []}}. Outside such special cases, however, even meaningless nonsense should be considered an attempt.

CRITICAL: You MUST wrap your JSON output in <json></json> tags exactly as shown in the example above."""

    async def grade_response(
        self, response: str, prompt: str, feature_label: str
    ) -> Dict:
        """
        Grade a response using Claude.

        Args:
            response: The model's response to grade
            prompt: The original user prompt
            feature_label: Label of the feature that was boosted (for context)

        Returns:
            Dictionary with grading results including 'raw_response' and parsed scores
        """
        # Rate limit
        await self.rate_limiter.acquire()

        # Format the judge prompt
        user_message = (
            f"Prompt: {prompt}\n"
            f"Response: {response}\n"
            f"Unrelated feature: {feature_label}"
        )

        raw_response = None
        try:
            # Call Claude API
            message = await self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": user_message},
                ],
                timeout=self.timeout,
            )

            if not message.content:
                return {
                    "raw_response": None,
                    "error": f"Claude API returned empty content (stop_reason={message.stop_reason}, usage={message.usage})"
                }

            raw_response = message.content[0].text

            # Extract grade
            grade_data = extract_claude_grade(raw_response)
            return {"raw_response": raw_response, **grade_data}

        except anthropic.RateLimitError as e:
            return {"raw_response": raw_response, "error": f"Rate limit error: {str(e)}"}
        except anthropic.APIConnectionError as e:
            return {"raw_response": raw_response, "error": f"API connection error: {str(e)}"}
        except anthropic.APIStatusError as e:
            return {"raw_response": raw_response, "error": f"API status error ({e.status_code}): {str(e)}"}
        except Exception as e:
            # Return raw_response even if parsing failed, for debugging
            import traceback
            return {"raw_response": raw_response, "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}
