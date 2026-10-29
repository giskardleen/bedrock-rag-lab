"""Validation helpers for simulated Amazon Bedrock Converse responses."""

import json


_FAQ_EXTRACTION_SYSTEM_PROMPT = (
    "Extract frequently asked questions and their answers "
    "from the provided document. "
    "Return only a valid JSON object with a 'records' array. "
    "Each record must contain exactly these string fields: "
    "'question', 'answer', and 'category'. "
    "Do not include a 'source' field or any additional fields. "
    "Do not include Markdown fences or explanatory text. "
    'If no FAQs can be extracted, return {"records": []}.'
)

_STRUCTURED_FAQ_EXTRACTION_SYSTEM_PROMPT = (
    "Extract frequently asked questions and their answers "
    "from the provided document. "
    "Populate each record with a question, answer, and category. "
    "Do not include source provenance; the application assigns it. "
    "If no FAQs can be extracted, return an empty records array."
)

_FAQ_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["question", "answer", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}

CLAUDE_HAIKU_45_INFERENCE_PROFILE = (
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)


def build_converse_request(
    document_text: str,
    model_id: str = "amazon.nova-micro-v1:0",
) -> dict:
    """Build arguments for a future Bedrock Converse call without side effects."""
    if not isinstance(document_text, str):
        raise TypeError("document_text must be a string")
    if not document_text.strip():
        raise ValueError("document_text must not be blank")

    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    if not model_id.strip():
        raise ValueError("model_id must not be blank")

    return {
        "modelId": model_id,
        "system": [{"text": _FAQ_EXTRACTION_SYSTEM_PROMPT}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": document_text}],
            }
        ],
        "inferenceConfig": {
            "temperature": 0,
            "maxTokens": 1024,
        },
    }


def build_structured_converse_request(
    document_text: str,
    model_id: str = CLAUDE_HAIKU_45_INFERENCE_PROFILE,
) -> dict:
    """Build a Converse request constrained by the FAQ extraction JSON Schema."""
    request = build_converse_request(document_text, model_id=model_id)
    request["system"] = [{"text": _STRUCTURED_FAQ_EXTRACTION_SYSTEM_PROMPT}]
    request["outputConfig"] = {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": json.dumps(
                        _FAQ_EXTRACTION_JSON_SCHEMA,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                }
            },
        }
    }
    return request


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
