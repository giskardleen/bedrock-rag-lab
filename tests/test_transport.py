"""Offline tests for the Bedrock Runtime transport boundary."""

import pytest
from bedrock_rag.transport import invoke_converse

class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def converse(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return self.response


def test_t01_forwards_request_fields_unchanged() -> None:
    response = {"output": {"message": {"content": [{"text": "result"}]}}}
    client = FakeClient(response)
    request = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "document"}]}],
        "inferenceConfig": {"temperature": 0, "maxTokens": 1024},
    }

    invoke_converse(client, request)

    assert client.calls == [request]


def test_t02_returns_exact_raw_response_object() -> None:
    response = {"output": {"message": {"content": [{"text": "result"}]}}}
    client = FakeClient(response)

    result = invoke_converse(client, {"modelId": "model-id"})

    assert result is response


def test_t03_propagates_original_exception() -> None:
    error = RuntimeError("transport failure")

    class FailingClient:
        def converse(self, **kwargs: object) -> dict:
            raise error

    with pytest.raises(RuntimeError) as captured:
        invoke_converse(FailingClient(), {"modelId": "model-id"})

    assert captured.value is error


def test_t04_invokes_converse_exactly_once() -> None:
    client = FakeClient({"status": "ok"})

    invoke_converse(client, {"modelId": "model-id"})

    assert len(client.calls) == 1
