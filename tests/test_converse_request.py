"""Offline tests for the Bedrock Converse request builder."""

import json

import pytest

from bedrock_rag.converse import (
    CLAUDE_HAIKU_45_INFERENCE_PROFILE,
    build_converse_request,
    build_structured_converse_request,
)


EXPECTED_SYSTEM_PROMPT = (
    "Extract frequently asked questions and their answers "
    "from the provided document. "
    "Return only a valid JSON object with a 'records' array. "
    "Each record must contain exactly these string fields: "
    "'question', 'answer', and 'category'. "
    "Do not include a 'source' field or any additional fields. "
    "Do not include Markdown fences or explanatory text. "
    'If no FAQs can be extracted, return {"records": []}.'
)


def test_r01_returns_complete_request_structure() -> None:
    request = build_converse_request("Document content")

    assert request == {
        "modelId": "amazon.nova-micro-v1:0",
        "system": [{"text": EXPECTED_SYSTEM_PROMPT}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Document content"}],
            }
        ],
        "inferenceConfig": {
            "temperature": 0,
            "maxTokens": 1024,
        },
    }


def test_r02_uses_default_model() -> None:
    request = build_converse_request("Document content")

    assert request["modelId"] == "amazon.nova-micro-v1:0"


def test_r03_accepts_custom_model_id() -> None:
    request = build_converse_request("Document content", model_id="provider.model-v1")

    assert request["modelId"] == "provider.model-v1"


def test_r04_separates_system_instructions_from_user_message() -> None:
    request = build_converse_request("Document content")

    assert request["system"] == [{"text": EXPECTED_SYSTEM_PROMPT}]
    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"text": "Document content"}],
        }
    ]


def test_r05_preserves_document_text_exactly() -> None:
    document_text = "  Document content with surrounding whitespace.\n"

    request = build_converse_request(document_text)

    assert request["messages"][0]["content"][0]["text"] == document_text


def test_r06_preserves_model_id_exactly() -> None:
    model_id = "  provider.custom-model-v1  "

    request = build_converse_request("Document content", model_id=model_id)

    assert request["modelId"] == model_id


def test_r07_uses_fixed_inference_configuration() -> None:
    request = build_converse_request("Document content")

    assert request["inferenceConfig"] == {
        "temperature": 0,
        "maxTokens": 1024,
    }


def test_r08_excludes_source_output_config_and_region() -> None:
    request = build_converse_request("Document content")

    assert set(request) == {"modelId", "system", "messages", "inferenceConfig"}
    assert set(request["messages"][0]) == {"role", "content"}
    assert set(request["messages"][0]["content"][0]) == {"text"}


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
@pytest.mark.parametrize("parameter", ["document_text", "model_id"])
def test_r09_rejects_incorrect_types_with_type_error(
    parameter: str,
    value: object,
) -> None:
    arguments = {
        "document_text": "Document content",
        "model_id": "amazon.nova-micro-v1:0",
    }
    arguments[parameter] = value

    with pytest.raises(TypeError):
        build_converse_request(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
@pytest.mark.parametrize("parameter", ["document_text", "model_id"])
def test_r10_rejects_blank_strings_with_value_error(
    parameter: str,
    value: str,
) -> None:
    arguments = {
        "document_text": "Document content",
        "model_id": "amazon.nova-micro-v1:0",
    }
    arguments[parameter] = value

    with pytest.raises(ValueError):
        build_converse_request(**arguments)


def test_r11_returns_independent_request_dictionaries() -> None:
    first = build_converse_request("Document content")
    second = build_converse_request("Document content")

    first["system"][0]["text"] = "Changed"
    first["messages"][0]["content"][0]["text"] = "Changed"
    first["inferenceConfig"]["maxTokens"] = 1

    assert second["system"] == [{"text": EXPECTED_SYSTEM_PROMPT}]
    assert second["messages"][0]["content"][0]["text"] == "Document content"
    assert second["inferenceConfig"]["maxTokens"] == 1024


def test_structured_request_uses_claude_profile_and_output_config() -> None:
    request = build_structured_converse_request("Document content")

    assert request["modelId"] == CLAUDE_HAIKU_45_INFERENCE_PROFILE
    assert request["inferenceConfig"] == {"temperature": 0, "maxTokens": 1024}
    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"text": "Document content"}],
        }
    ]
    assert request["outputConfig"]["textFormat"]["type"] == "json_schema"


def test_structured_request_schema_matches_raw_parser_contract() -> None:
    request = build_structured_converse_request("Document content")
    schema_text = request["outputConfig"]["textFormat"]["structure"][
        "jsonSchema"
    ]["schema"]

    assert isinstance(schema_text, str)
    schema = json.loads(schema_text)
    assert schema == {
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
    assert "source" not in schema_text
    assert "minLength" not in schema_text


def test_structured_prompt_defers_format_and_provenance_to_contracts() -> None:
    request = build_structured_converse_request("Document content")
    prompt = request["system"][0]["text"]

    assert "question, answer, and category" in prompt
    assert "application assigns" in prompt
    assert "JSON" not in prompt
    assert "Markdown" not in prompt


def test_structured_request_accepts_custom_model_and_returns_independent_data() -> None:
    first = build_structured_converse_request(
        "Document content",
        model_id="profile.custom-v1",
    )
    second = build_structured_converse_request("Document content")

    first["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"] = "{}"

    assert first["modelId"] == "profile.custom-v1"
    assert second["modelId"] == CLAUDE_HAIKU_45_INFERENCE_PROFILE
    assert json.loads(
        second["outputConfig"]["textFormat"]["structure"]["jsonSchema"][
            "schema"
        ]
    )["required"] == ["records"]
