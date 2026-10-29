"""Manually invoked, opt-in live Amazon Bedrock Converse smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from bedrock_rag.converse import (
    build_converse_request,
    extract_converse_text,
)
from bedrock_rag.parsing import parse_faq_extraction
from bedrock_rag.transport import invoke_converse


REGION = "us-east-1"
MODEL_ID = "amazon.nova-lite-v1:0"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "python_general_faq_excerpt.txt"
)
FIXTURE_SHA256 = "316531283c9a537e496d12552977d24c5810b0352c5655501a80763ae4d1b033"
SOURCE = "https://docs.python.org/3.11/faq/general.html"
ALLOWED_STOP_REASONS = {
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "guardrail_intervened",
    "content_filtered",
    "malformed_model_output",
    "malformed_tool_use",
    "model_context_window_exceeded",
}


def _load_verified_fixture(path: Path | None = None) -> str:
    if path is None:
        path = FIXTURE_PATH
    fixture_bytes = path.read_bytes()
    actual_hash = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_hash != FIXTURE_SHA256:
        raise ValueError("fixture SHA-256 verification failed")
    return fixture_bytes.decode("utf-8")


def _print_response_metadata(response: dict[str, Any]) -> None:
    stop_reason = response.get("stopReason")
    if isinstance(stop_reason, str) and stop_reason in ALLOWED_STOP_REASONS:
        print(f"stop_reason={stop_reason}")
    else:
        print("stop_reason=unknown")

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    for key, label in (
        ("inputTokens", "input_tokens"),
        ("outputTokens", "output_tokens"),
        ("totalTokens", "total_tokens"),
    ):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            print(f"{label}={value}")


def _classify_sdk_failure(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        error_data = response.get("Error")
        if isinstance(error_data, dict):
            code = error_data.get("Code")
            if code in {
                "InvalidClientTokenId",
                "InvalidSignatureException",
                "UnrecognizedClientException",
            }:
                return "authentication"
            if code in {
                "AccessDeniedException",
                "UnauthorizedException",
            }:
                return "authorization"
    return "transport"


def _run_live(boto3_module: Any, config_class: Any, document_text: str) -> int:
    try:
        config = config_class(
            retries={"total_max_attempts": 1, "mode": "standard"}
        )
        client = boto3_module.client(
            "bedrock-runtime",
            region_name=REGION,
            config=config,
        )
    except Exception as error:
        print("invocation_attempted=false")
        print(f"failure_stage={_classify_sdk_failure(error)}")
        return 1

    try:
        request = build_converse_request(document_text, model_id=MODEL_ID)
    except Exception:
        print("invocation_attempted=false")
        print("failure_stage=request_construction")
        return 1

    print("invocation_attempted=true")
    try:
        response = invoke_converse(client, request)
    except Exception as error:
        print("converse_response_received=false")
        print(f"failure_stage={_classify_sdk_failure(error)}")
        return 1

    print("converse_response_received=true")
    if not isinstance(response, dict):
        print("response_extraction_succeeded=false")
        print("schema_validation_succeeded=false")
        print("failure_stage=response_contract")
        return 1

    _print_response_metadata(response)

    try:
        text = extract_converse_text(response)
    except Exception:
        print("response_extraction_succeeded=false")
        print("schema_validation_succeeded=false")
        print("failure_stage=response_contract")
        return 1

    print("response_extraction_succeeded=true")
    try:
        extraction = parse_faq_extraction(text, source=SOURCE)
    except json.JSONDecodeError:
        print("schema_validation_succeeded=false")
        print("failure_stage=schema_validation")
        print("failure_detail=invalid_json")
        return 1
    except ValidationError:
        print("schema_validation_succeeded=false")
        print("failure_stage=schema_validation")
        print("failure_detail=schema_incompatible")
        return 1
    except Exception:
        print("schema_validation_succeeded=false")
        print("failure_stage=schema_validation")
        print("failure_detail=unexpected_parser_error")
        return 1

    print("schema_validation_succeeded=true")
    print(f"faq_entries_extracted={len(extraction.records)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="explicitly permit one live Amazon Bedrock Converse request",
    )
    args = parser.parse_args(argv)

    if not args.execute_live:
        print("invocation_attempted=false")
        print("live_execution_enabled=false")
        return 0

    try:
        document_text = _load_verified_fixture()
    except (OSError, UnicodeDecodeError, ValueError):
        print("invocation_attempted=false")
        print("failure_stage=fixture_validation")
        return 1

    try:
        boto3_module = importlib.import_module("boto3")
        config_module = importlib.import_module("botocore.config")
    except ImportError:
        print("invocation_attempted=false")
        print("failure_stage=live_dependency")
        return 1

    return _run_live(boto3_module, config_module.Config, document_text)


if __name__ == "__main__":
    raise SystemExit(main())
