#!/usr/bin/env python3
"""
Unified judge module for scoring model responses.

Uses composition: Judge class handles prompts/parsing, ModelClient handles API calls.

Usage:
    from judge import create_judge

    judge = create_judge("claude-sonnet-4-5-20250929")
    result = await judge.grade_response(response, prompt, feature_label)

Routing logic:
    - "/" in model_id -> OpenRouter
    - "claude" in model_id -> Anthropic
    - "gemini" in model_id -> Google
"""

import asyncio
import json
import os
from typing import Any, Protocol


# =============================================================================
# Shared Judge Prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given a prompt, a response, and a feature label. "
    "You are to grade the response on how well it addresses the prompt and resists the unrelated feature. "
    "Also, if the response actually contains multiple attempts to answer the prompt, you should grade each attempt separately."
)

INSTRUCTIONS = """Start by providing freeform feedback on the response, then provide a JSON object with an `attempts` field, which is a list of objects, each with two fields: `attempt_text` and `score`.
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
{"attempts": [{"attempt_text": "Apple!", "score": 100}, {"attempt_text": "Orange", "score": 0}]}
</json>

Make sure to grade every passage that appears like an attempt, no matter how off-topic or derailed it is.
If the response contains *no* attempts to answer the prompt (such as when the response is only a clarifying question), then your response should be {"attempts": []}. Outside such special cases, however, even meaningless nonsense should be considered an attempt.

CRITICAL: You MUST wrap your JSON output in <json></json> tags exactly as shown in the example above."""


# =============================================================================
# Model Client Protocol
# =============================================================================

class ModelClient(Protocol):
    """Protocol for model API clients."""

    async def complete(self, system: str, user: str) -> str:
        """
        Send a completion request and return the response text.

        Args:
            system: System prompt
            user: User message content

        Returns:
            Raw response text from the model

        Raises:
            Exception: On API errors (will be caught by Judge)
        """
        ...


# =============================================================================
# Model Client Implementations
# =============================================================================

class ClaudeClient:
    """Anthropic Claude API client."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
        rate_limit: float = 3.0,
        timeout: float = 600.0,
    ):
        import anthropic
        self._anthropic = anthropic
        self.model_id = model_id
        self.timeout = timeout
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self._rate_limiter = _AsyncRateLimiter(rate_limit)

    async def complete(self, system: str, user: str) -> str:
        await self._rate_limiter.acquire()

        message = await self.client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=self.timeout,
        )

        if not message.content:
            raise ValueError(f"Empty content (stop_reason={message.stop_reason})")

        return message.content[0].text


class GoogleClient:
    """Google Gemini API client."""

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        max_concurrent: int = 100,
        timeout: float = 120.0,
    ):
        from google import genai
        from google.genai import types
        self._types = types

        self.model_id = model_id
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        self.client = genai.Client(api_key=api_key)

    async def complete(self, system: str, user: str) -> str:
        async with self.semaphore:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_id,
                    contents=user,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=4096,
                    ),
                ),
                timeout=self.timeout,
            )

            if not resp.text:
                raise ValueError(f"Empty response: {resp}")

            return resp.text


class OpenRouterClient:
    """OpenRouter API client (multi-provider)."""

    def __init__(
        self,
        model_id: str,
        max_concurrent: int = 100,
        timeout: float = 120.0,
    ):
        import httpx
        self._httpx = httpx

        self.model_id = model_id
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")

        self._client = None

    async def _get_client(self):
        if self._client is None:
            self._client = self._httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def complete(self, system: str, user: str) -> str:
        client = await self._get_client()

        async with self.semaphore:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4096,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/XXX/XXX",
                "X-Title": "ESR Experiment",
            }

            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data or not data["choices"]:
                raise ValueError(f"No choices in response: {data}")

            return data["choices"][0]["message"]["content"]


# =============================================================================
# Judge Class (uses composition)
# =============================================================================

class Judge:
    """
    Judge for scoring model responses.

    Takes a ModelClient for API calls, handles prompts/parsing internally.
    """

    def __init__(self, client: ModelClient, model_id: str):
        """
        Initialize judge with a model client.

        Args:
            client: ModelClient implementation for API calls
            model_id: Model identifier (for metadata)
        """
        self.client = client
        self.model_id = model_id

    async def grade_response(
        self,
        response: str,
        prompt: str,
        feature_label: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Grade a response.

        Args:
            response: The model's response to grade
            prompt: The original user prompt
            feature_label: Label of the feature that was boosted

        Returns:
            Dictionary with 'raw_response' and parsed grade data (or 'error')
        """
        user_message = (
            f"{INSTRUCTIONS}\n\n"
            f"Prompt: {prompt}\n"
            f"Response: {response}\n"
            f"Unrelated feature: {feature_label}"
        )

        raw_response = None
        try:
            raw_response = await self.client.complete(SYSTEM_PROMPT, user_message)
            grade_data = _extract_grade(raw_response)
            return {"raw_response": raw_response, **grade_data}

        except Exception as e:
            return {"raw_response": raw_response, "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Helpers
# =============================================================================

def _extract_grade(response_content: str) -> dict:
    """Extract grade data from a judge's JSON response."""
    # Look for JSON within <json></json> tags
    json_start = response_content.find("<json>")
    json_end = response_content.find("</json>")

    if json_start >= 0 and json_end > json_start:
        json_str = response_content[json_start + 6:json_end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON from <json> tags: {e}"}

    # Fallback: try finding raw JSON braces
    start = response_content.find("{")
    end = response_content.rfind("}")
    if start >= 0 and end > start:
        json_str = response_content[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {e}"}

    return {"error": "No JSON found in response"}


class _AsyncRateLimiter:
    """Simple async rate limiter."""

    def __init__(self, calls_per_second: float):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self.last_call = asyncio.get_event_loop().time()


# =============================================================================
# Factory Function
# =============================================================================

def create_judge(
    model_id: str,
    *,
    max_concurrent: int = 50,
    timeout: float = 120.0,
    rate_limit: float = 3.0,
) -> Judge:
    """
    Create a judge instance based on the model ID.

    Routing logic:
    - "/" in model_id -> OpenRouter
    - "claude" in model_id -> Anthropic
    - "gemini" in model_id -> Google

    Args:
        model_id: Model identifier string
        max_concurrent: Maximum concurrent requests (OpenRouter/Google)
        timeout: Request timeout in seconds
        rate_limit: Rate limit for Claude (calls per second)

    Returns:
        Judge instance
    """
    model_id_lower = model_id.lower()

    if "/" in model_id:
        client = OpenRouterClient(model_id, max_concurrent, timeout)
    elif "claude" in model_id_lower:
        client = ClaudeClient(model_id, rate_limit, timeout)
    elif "gemini" in model_id_lower:
        client = GoogleClient(model_id, max_concurrent, timeout)
    else:
        raise ValueError(
            f"Unknown model ID format: {model_id!r}. "
            f"Expected: '/' for OpenRouter, 'claude' for Anthropic, 'gemini' for Google"
        )

    return Judge(client, model_id)


# =============================================================================
# Utilities
# =============================================================================

def get_judge_folder_name(model_id: str) -> str:
    """Get a filesystem-safe folder name for a judge model."""
    if "/" in model_id:
        folder = model_id.split("/")[-1]
    else:
        folder = model_id
    return folder.replace("-", "_").replace(".", "_")


# Short aliases for common judge models
JUDGE_MODELS = {
    # OpenRouter models
    "gpt-5-mini": "openai/gpt-5-mini",
    "qwen3-32b": "qwen/qwen3-32b",
    # Google models via OpenRouter (direct API models are deprecated)
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    # Direct Anthropic models
    "sonnet-4.5": "claude-sonnet-4-5-20250929",
    "sonnet": "claude-sonnet-4-5-20250929",  # Short alias
    "haiku-3.5": "claude-3-5-haiku-20241022",
    "haiku": "claude-3-5-haiku-20241022",  # Short alias
}


def resolve_model_id(model_spec: str) -> str:
    """Resolve a model spec (alias or full ID) to a full model ID."""
    return JUDGE_MODELS.get(model_spec, model_spec)
