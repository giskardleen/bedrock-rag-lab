"""Offline unit tests for Stage 1A data contracts and strict parsing."""

import json

import pytest
from pydantic import ValidationError

from bedrock_rag.parsing import parse_faq_extraction
from bedrock_rag.schemas import FAQExtraction, FAQRecord


def test_faq_record_accepts_valid_strict_strings_without_normalizing() -> None:
    record = FAQRecord(
        question="  When does support open?  ",
        answer="At 09:00.",
        category="support",
        source="docs/support.md",
    )

    assert record.question == "  When does support open?  "


@pytest.mark.parametrize("field", ["question", "answer", "category", "source"])
@pytest.mark.parametrize("value", ["", " \t\n "])
def test_faq_record_rejects_blank_text(field: str, value: str) -> None:
    data = {
        "question": "Question?",
        "answer": "Answer.",
        "category": "general",
        "source": "docs/faq.md",
    }
    data[field] = value

    with pytest.raises(ValidationError):
        FAQRecord.model_validate(data)


@pytest.mark.parametrize("value", [1, 1.5, True, None, ["text"], {"text": "value"}])
def test_faq_record_rejects_non_string_text(value: object) -> None:
    with pytest.raises(ValidationError):
        FAQRecord.model_validate(
            {
                "question": value,
                "answer": "Answer.",
                "category": "general",
                "source": "docs/faq.md",
            }
        )


def test_faq_record_rejects_missing_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FAQRecord.model_validate(
            {
                "question": "Question?",
                "answer": "Answer.",
                "category": "general",
            }
        )

    with pytest.raises(ValidationError):
        FAQRecord.model_validate(
            {
                "question": "Question?",
                "answer": "Answer.",
                "category": "general",
                "source": "docs/faq.md",
                "unexpected": "value",
            }
        )


def test_faq_extraction_accepts_and_preserves_empty_records() -> None:
    extraction = FAQExtraction.model_validate({"records": []})

    assert extraction.records == []


def test_parser_attaches_application_source_to_multiple_records() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "How do I reset my password?",
                    "answer": "Use the account recovery page.",
                    "category": "account",
                },
                {
                    "question": "When is support available?",
                    "answer": "Support is available on weekdays.",
                    "category": "support",
                },
            ]
        }
    )

    extraction = parse_faq_extraction(payload, source="docs/help.md")

    assert len(extraction.records) == 2
    assert {record.source for record in extraction.records} == {"docs/help.md"}


def test_parser_preserves_empty_extraction() -> None:
    extraction = parse_faq_extraction('{"records": []}', source="docs/empty.md")

    assert extraction.records == []


def test_parser_distinguishes_invalid_json_syntax() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_faq_extraction('{"records": [}', source="docs/faq.md")


@pytest.mark.parametrize(
    "payload",
    [
        '{"records": [{"answer": "Answer.", "category": "general"}]}',
        '{"records": [{"question": "Question?", "answer": "Answer.", '
        '"category": "general", "unexpected": "value"}]}',
        '{"records": [{"question": 42, "answer": "Answer.", '
        '"category": "general"}]}',
        '{"records": [{"question": " ", "answer": "Answer.", '
        '"category": "general"}]}',
        '{"records": "not-a-list"}',
        '{"records": [], "unexpected": true}',
    ],
)
def test_parser_distinguishes_schema_validation_errors(payload: str) -> None:
    with pytest.raises(ValidationError):
        parse_faq_extraction(payload, source="docs/faq.md")


def test_parser_rejects_source_supplied_by_model() -> None:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "Question?",
                    "answer": "Answer.",
                    "category": "general",
                    "source": "untrusted/model/value",
                }
            ]
        }
    )

    with pytest.raises(ValidationError):
        parse_faq_extraction(payload, source="docs/trusted.md")


@pytest.mark.parametrize("source", ["", " \t ", 123, None])
def test_parser_rejects_invalid_application_source(source: object) -> None:
    with pytest.raises(ValidationError):
        parse_faq_extraction('{"records": []}', source=source)  # type: ignore[arg-type]
