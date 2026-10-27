"""Validation helpers for simulated Amazon Bedrock Converse responses."""


class ConverseResponseError(Exception):
    """Raised when a Converse response violates the adapter contract."""


def extract_converse_text(response: dict) -> str:
    """Validate a Converse response and return its single text block unchanged."""
    if not isinstance(response, dict):
        raise ConverseResponseError("response must be a dictionary")

    if response.get("stopReason") != "end_turn":
        raise ConverseResponseError("stopReason must be 'end_turn'")

    output = response.get("output")
    if not isinstance(output, dict):
        raise ConverseResponseError("output must be a dictionary")

    message = output.get("message")
    if not isinstance(message, dict):
        raise ConverseResponseError("output.message must be a dictionary")

    content = message.get("content")
    if not isinstance(content, list):
        raise ConverseResponseError("output.message.content must be a list")

    if len(content) != 1:
        raise ConverseResponseError(
            "output.message.content must contain exactly one element"
        )

    block = content[0]
    if not isinstance(block, dict):
        raise ConverseResponseError("the content element must be a dictionary")

    text = block.get("text")
    if not isinstance(text, str):
        raise ConverseResponseError("the content text must be a string")

    if not text.strip():
        raise ConverseResponseError("the content text must not be blank")

    return text
