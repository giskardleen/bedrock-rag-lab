"""Offline tests for the protected Unit 05 live smoke script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "labs"
    / "stage1b_unit05_live_converse_smoke.py"
)


@pytest.fixture
def smoke_module() -> Any:
    spec = importlib.util.spec_from_file_location("unit05_live_smoke", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _successful_response() -> dict[str, Any]:
    payload = json.dumps(
        {
            "records": [
                {
                    "question": "What is Python?",
                    "answer": "A programming language.",
                    "category": "general",
                },
                {
                    "question": "What is the Python Software Foundation?",
                    "answer": "A non-profit organization.",
                    "category": "general",
                },
            ]
        }
    )
    return {
        "stopReason": "end_turn",
        "usage": {"inputTokens": 20, "outputTokens": 30, "totalTokens": 50},
        "output": {"message": {"content": [{"text": payload}]}},
    }


class FakeConfig:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


class FakeBoto3:
    def __init__(self, client: Any) -> None:
        self.fake_client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def client(self, service_name: str, **kwargs: Any) -> Any:
        self.calls.append((service_name, kwargs))
        return self.fake_client


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, module: Any, client: Any) -> FakeBoto3:
    fake_boto3 = FakeBoto3(client)

    def fake_import(name: str) -> Any:
        if name == "boto3":
            return fake_boto3
        if name == "botocore.config":
            return SimpleNamespace(Config=FakeConfig)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    return fake_boto3


def test_without_opt_in_stops_before_boto_import_or_client_creation(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_import(name: str) -> Any:
        if name in {"boto3", "botocore.config"}:
            raise AssertionError("AWS SDK import was attempted")
        return __import__(name)

    monkeypatch.setattr(smoke_module.importlib, "import_module", forbidden_import)

    assert smoke_module.main([]) == 0
    assert capsys.readouterr().out == (
        "invocation_attempted=false\nlive_execution_enabled=false\n"
    )


def test_invalid_fixture_hash_prevents_invocation(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_fixture = tmp_path / "fixture.txt"
    invalid_fixture.write_text("changed", encoding="utf-8")
    monkeypatch.setattr(smoke_module, "FIXTURE_PATH", invalid_fixture)
    monkeypatch.setattr(
        smoke_module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected SDK import: {name}"),
    )

    assert smoke_module.main(["--execute-live"]) == 1
    assert capsys.readouterr().out == (
        "invocation_attempted=false\nfailure_stage=fixture_validation\n"
    )


def test_model_region_retries_and_existing_pipeline_are_used(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeConfig.calls.clear()
    client = FakeClient(_successful_response())
    fake_boto3 = _install_fake_sdk(monkeypatch, smoke_module, client)
    calls: list[tuple[str, Any]] = []

    original_builder = smoke_module.build_converse_request
    original_transport = smoke_module.invoke_converse
    original_adapter = smoke_module.extract_converse_text
    original_parser = smoke_module.parse_faq_extraction

    def builder(document_text: str, model_id: str) -> dict[str, Any]:
        calls.append(("builder", model_id))
        return original_builder(document_text, model_id=model_id)

    def transport(client_arg: Any, request: dict[str, Any]) -> dict[str, Any]:
        calls.append(("transport", request))
        return original_transport(client_arg, request)

    def adapter(response: dict[str, Any]) -> str:
        calls.append(("adapter", response))
        return original_adapter(response)

    def parser(payload: str, *, source: str) -> Any:
        calls.append(("parser", source))
        return original_parser(payload, source=source)

    monkeypatch.setattr(smoke_module, "build_converse_request", builder)
    monkeypatch.setattr(smoke_module, "invoke_converse", transport)
    monkeypatch.setattr(smoke_module, "extract_converse_text", adapter)
    monkeypatch.setattr(smoke_module, "parse_faq_extraction", parser)

    assert smoke_module.main(["--execute-live"]) == 0
    assert [name for name, _ in calls] == [
        "builder",
        "transport",
        "adapter",
        "parser",
    ]
    assert calls[0] == ("builder", "amazon.nova-lite-v1:0")
    assert fake_boto3.calls[0][0] == "bedrock-runtime"
    assert fake_boto3.calls[0][1]["region_name"] == "us-east-1"
    assert FakeConfig.calls == [
        {"retries": {"total_max_attempts": 1, "mode": "standard"}}
    ]
    assert len(client.calls) == 1


def test_mocked_success_prints_only_sanitized_evidence(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = _successful_response()
    _install_fake_sdk(monkeypatch, smoke_module, FakeClient(response))

    assert smoke_module.main(["--execute-live"]) == 0
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=true\n"
        "converse_response_received=true\n"
        "stop_reason=end_turn\n"
        "input_tokens=20\n"
        "output_tokens=30\n"
        "total_tokens=50\n"
        "response_extraction_succeeded=true\n"
        "schema_validation_succeeded=true\n"
        "faq_entries_extracted=2\n"
    )
    assert response["output"]["message"]["content"][0]["text"] not in output
    assert "A programming language." not in output


def test_mocked_failure_does_not_print_secrets_or_raw_content(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("SECRET_ACCESS_KEY raw-response-content")

    _install_fake_sdk(monkeypatch, smoke_module, FailingClient())

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=true\n"
        "converse_response_received=false\n"
        "failure_stage=transport\n"
    )
    assert "SECRET_ACCESS_KEY" not in output
    assert "raw-response-content" not in output


def test_non_dictionary_response_is_a_sanitized_contract_failure(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_response = "SECRET raw response"
    _install_fake_sdk(monkeypatch, smoke_module, FakeClient(raw_response))

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=true\n"
        "converse_response_received=true\n"
        "response_extraction_succeeded=false\n"
        "schema_validation_succeeded=false\n"
        "failure_stage=response_contract\n"
    )
    assert raw_response not in output


@pytest.mark.parametrize(
    "stop_reason",
    ["SECRET_STOP_REASON", {"secret": "raw-metadata"}],
)
def test_unexpected_stop_reason_is_reported_as_unknown_without_disclosure(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop_reason: object,
) -> None:
    response = _successful_response()
    response["stopReason"] = stop_reason
    _install_fake_sdk(monkeypatch, smoke_module, FakeClient(response))

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert "stop_reason=unknown\n" in output
    assert "SECRET_STOP_REASON" not in output
    assert "raw-metadata" not in output
    assert "failure_stage=response_contract\n" in output


@pytest.mark.parametrize("missing_dependency", ["boto3", "botocore.config"])
def test_missing_live_dependency_stops_before_client_construction(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing_dependency: str,
) -> None:
    fake_boto3 = FakeBoto3(pytest.fail)

    def fake_import(name: str) -> Any:
        if name == missing_dependency:
            raise ImportError("SECRET dependency detail")
        if name == "boto3":
            return fake_boto3
        return SimpleNamespace(Config=FakeConfig)

    monkeypatch.setattr(smoke_module.importlib, "import_module", fake_import)

    assert smoke_module.main(["--execute-live"]) == 1
    assert fake_boto3.calls == []
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=false\nfailure_stage=live_dependency\n"
    )
    assert "SECRET" not in output


def test_client_construction_failure_is_sanitized(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingBoto3:
        def client(self, service_name: str, **kwargs: Any) -> Any:
            raise RuntimeError("SECRET client construction detail")

    def fake_import(name: str) -> Any:
        if name == "boto3":
            return FailingBoto3()
        return SimpleNamespace(Config=FakeConfig)

    monkeypatch.setattr(smoke_module.importlib, "import_module", fake_import)

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert output == "invocation_attempted=false\nfailure_stage=transport\n"
    assert "SECRET" not in output


def test_authorization_failure_is_classified_without_disclosure(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class AuthorizationError(Exception):
        response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "SECRET authorization detail",
            }
        }

    class DeniedClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise AuthorizationError("SECRET raw exception")

    _install_fake_sdk(monkeypatch, smoke_module, DeniedClient())

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=true\n"
        "converse_response_received=false\n"
        "failure_stage=authorization\n"
    )
    assert "SECRET" not in output


@pytest.mark.parametrize(
    "payload",
    [
        "SECRET invalid JSON {",
        json.dumps(
            {
                "records": [
                    {
                        "answer": "SECRET schema payload",
                        "category": "general",
                    }
                ]
            }
        ),
    ],
)
def test_model_payload_failures_are_sanitized_schema_failures(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: str,
) -> None:
    response = _successful_response()
    response["output"]["message"]["content"] = [{"text": payload}]
    _install_fake_sdk(monkeypatch, smoke_module, FakeClient(response))

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert "response_extraction_succeeded=true\n" in output
    assert "schema_validation_succeeded=false\n" in output
    assert "failure_stage=schema_validation\n" in output
    assert "SECRET" not in output
    assert payload not in output


def test_unexpected_request_construction_exception_is_sanitized(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = FakeClient(_successful_response())
    _install_fake_sdk(monkeypatch, smoke_module, client)
    monkeypatch.setattr(
        smoke_module,
        "build_converse_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SECRET request payload")
        ),
    )

    assert smoke_module.main(["--execute-live"]) == 1
    assert client.calls == []
    output = capsys.readouterr().out
    assert output == (
        "invocation_attempted=false\nfailure_stage=request_construction\n"
    )
    assert "SECRET" not in output


def test_unexpected_response_adaptation_exception_is_sanitized(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_sdk(
        monkeypatch,
        smoke_module,
        FakeClient(_successful_response()),
    )
    monkeypatch.setattr(
        smoke_module,
        "extract_converse_text",
        lambda response: (_ for _ in ()).throw(
            RuntimeError("SECRET raw response")
        ),
    )

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert "response_extraction_succeeded=false\n" in output
    assert "schema_validation_succeeded=false\n" in output
    assert "failure_stage=response_contract\n" in output
    assert "SECRET" not in output


def test_unexpected_parsing_exception_is_sanitized(
    smoke_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_sdk(
        monkeypatch,
        smoke_module,
        FakeClient(_successful_response()),
    )
    monkeypatch.setattr(
        smoke_module,
        "parse_faq_extraction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SECRET model payload")
        ),
    )

    assert smoke_module.main(["--execute-live"]) == 1
    output = capsys.readouterr().out
    assert "response_extraction_succeeded=true\n" in output
    assert "schema_validation_succeeded=false\n" in output
    assert "failure_stage=schema_validation\n" in output
    assert "SECRET" not in output
