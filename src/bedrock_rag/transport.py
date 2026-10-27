"""Minimal transport for an injected Bedrock Runtime client."""

from typing import Any


def invoke_converse(
    client: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Invoke Bedrock Converse and return its raw response."""
    return client.converse(**request)
