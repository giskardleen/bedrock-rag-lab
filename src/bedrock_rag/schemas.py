"""Domain contracts for FAQ knowledge extracted from source documents."""

from typing import Annotated
from pydantic import AfterValidator, BaseModel, ConfigDict, StrictStr

def _reject_blank(value: str) -> str:
    """Reject empty or whitespace-only text without normalizing valid input."""
    if not value.strip():
        raise ValueError("must not be empty or contain only whitespace")
    return value

NonBlankStrictStr = Annotated[StrictStr, AfterValidator(_reject_blank)]


class FAQRecord(BaseModel):
    """A validated knowledge record with application-controlled provenance."""

    model_config = ConfigDict(extra="forbid")

    question: NonBlankStrictStr
    answer: NonBlankStrictStr
    category: NonBlankStrictStr
    source: NonBlankStrictStr


class FAQExtraction(BaseModel):
    """The records extracted from one source; an empty result is valid."""

    model_config = ConfigDict(extra="forbid")

    records: list[FAQRecord]
