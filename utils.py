"""Utility functions for experiment."""

import json


def extract_claude_grade(response_content: str) -> dict:
    """
    Extract the grade data from Claude's JSON response.

    Args:
        response_content: String content from Claude's response

    Returns:
        Dictionary with parsed JSON data

    Raises:
        ValueError: If no valid JSON found in response
    """
    # Look for JSON within <json></json> tags
    json_start = response_content.find("<json>")
    json_end = response_content.find("</json>")

    if json_start >= 0 and json_end > json_start:
        json_str = response_content[json_start + 6:json_end].strip()
        try:
            eval_data = json.loads(json_str)
            return eval_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from <json> tags. Error: {str(e)}. JSON preview: {json_str[:200]}...")

    # Fallback: try old method (for backwards compatibility)
    start = response_content.find("{")
    end = response_content.rfind("}")
    if start >= 0 and end > start:
        json_str = response_content[start : end + 1]
        try:
            eval_data = json.loads(json_str)
            return eval_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON. Error: {str(e)}. JSON preview: {json_str[:200]}...")

    raise ValueError("No JSON found in response (no <json> tags or braces)")
