"""Strict JSON parsing for FAQ extraction results."""

import json
from pydantic import BaseModel, ConfigDict, TypeAdapter
from bedrock_rag.schemas import FAQExtraction, NonBlankStrictStr

class _ExtractedFAQRecord(BaseModel):
    """LLM-facing record shape, intentionally excluding source provenance."""

    model_config = ConfigDict(extra="forbid")

    question: NonBlankStrictStr
    answer: NonBlankStrictStr
    category: NonBlankStrictStr

class _RawFAQExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[_ExtractedFAQRecord]

_source_adapter = TypeAdapter(NonBlankStrictStr)

def parse_faq_extraction(payload: str, *, source: str) -> FAQExtraction:
    """Parse model JSON and attach trusted, application-provided provenance.

    Invalid JSON raises ``json.JSONDecodeError``. A valid JSON document that
    violates either the extraction shape or the domain contract raises
    ``pydantic.ValidationError``.
    """
    decoded = json.loads(payload)
    extracted = _RawFAQExtraction.model_validate(decoded)
    validated_source = _source_adapter.validate_python(source)

    return FAQExtraction.model_validate(
        {
            "records": [
                {
                    "question": record.question,
                    "answer": record.answer,
                    "category": record.category,
                    "source": validated_source,
                }
                for record in extracted.records
            ]
        }
    )
