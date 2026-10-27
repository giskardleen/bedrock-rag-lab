"""Offline integration tests for Converse text extraction and FAQ parsing."""

import json

import pytest
from pydantic import ValidationError

from bedrock_rag.converse import ConverseResponseError, extract_converse_text
from bedrock_rag.parsing import parse_faq_extraction
from bedrock_rag.schemas import FAQExtraction


def converse_response(text: str, *, stop_reason: str = "end_turn") -> dict:
    return {
        "stopReason": stop_reason,
        "output": {
            "message": {
                "content": [{"text": text}],
            }
        },
    }


def test_i01_valid_converse_and_json_produce_validated_extraction() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "How can I contact support?",
                    "answer": "Use the support form.",
                    "category": "support",
                }
            ]
        }
    )

    text = extract_converse_text(converse_response(payload))
    extraction = parse_faq_extraction(text, source="docs/support.md")

    assert isinstance(extraction, FAQExtraction)
    assert extraction.records[0].question == "How can I contact support?"
    assert extraction.records[0].source == "docs/support.md"


def test_i02_max_tokens_raises_converse_response_error() -> None:
    response = converse_response('{"records": []}', stop_reason="max_tokens")

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_i03_malformed_json_raises_json_decode_error() -> None:
    text = extract_converse_text(converse_response('{"records": [}'))

    with pytest.raises(json.JSONDecodeError):
        parse_faq_extraction(text, source="docs/faq.md")


def test_i04_invalid_record_raises_validation_error() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "answer": "Use the support form.",
                    "category": "support",
                }
            ]
        }
    )
    text = extract_converse_text(converse_response(payload))

    with pytest.raises(ValidationError):
        parse_faq_extraction(text, source="docs/support.md")


def test_i05_multiple_records_receive_application_source() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "How can I contact support?",
                    "answer": "Use the support form.",
                    "category": "support",
                },
                {
                    "question": "When is support available?",
                    "answer": "Support is available on weekdays.",
                    "category": "support",
                },
            ]
        }
    )

    text = extract_converse_text(converse_response(payload))
    extraction = parse_faq_extraction(text, source="docs/help-center.md")

    assert len(extraction.records) == 2
    assert [record.source for record in extraction.records] == [
        "docs/help-center.md",
        "docs/help-center.md",
    ]


def test_i06_empty_records_collection_is_preserved() -> None:
    text = extract_converse_text(converse_response('{"records": []}'))

    extraction = parse_faq_extraction(text, source="docs/empty.md")

    assert extraction.records == []


def test_i07_model_supplied_source_raises_validation_error() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "How can I contact support?",
                    "answer": "Use the support form.",
                    "category": "support",
                    "source": "untrusted/model-source.md",
                }
            ]
        }
    )
    text = extract_converse_text(converse_response(payload))

    with pytest.raises(ValidationError):
        parse_faq_extraction(text, source="docs/trusted-source.md")


def test_i08_surrounding_spaces_are_preserved_and_json_is_accepted() -> None:
    payload = '  {"records": []}  '

    text = extract_converse_text(converse_response(payload))
    extraction = parse_faq_extraction(text, source="docs/spaced.md")

    assert text == payload
    assert extraction.records == []
