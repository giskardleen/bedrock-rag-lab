"""Unit tests for the strict Converse response adapter contract."""

import pytest

from bedrock_rag.converse import ConverseResponseError, extract_converse_text


def valid_response(text: str = "Extracted answer") -> dict:
    return {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "content": [{"text": text}],
            }
        },
    }


def test_b01_returns_valid_text_without_modification() -> None:
    text = "  Extracted answer with surrounding spaces.  "

    assert extract_converse_text(valid_response(text)) == text


def test_b02_rejects_max_tokens_stop_reason() -> None:
    response = valid_response()
    response["stopReason"] = "max_tokens"

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_b03_rejects_missing_stop_reason() -> None:
    response = valid_response()
    del response["stopReason"]

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_b04_rejects_missing_output_message() -> None:
    response = valid_response()
    del response["output"]["message"]

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_b05_rejects_empty_content() -> None:
    response = valid_response()
    response["output"]["message"]["content"] = []

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_b06_rejects_block_without_text() -> None:
    response = valid_response()
    response["output"]["message"]["content"] = [{}]

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


def test_b07_rejects_multiple_content_blocks() -> None:
    response = valid_response()
    response["output"]["message"]["content"].append({"text": "Second block"})

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_b08_rejects_empty_or_whitespace_only_text(text: str) -> None:
    with pytest.raises(ConverseResponseError):
        extract_converse_text(valid_response(text))


@pytest.mark.parametrize("response", [None, [], "response", 1])
def test_rejects_non_dictionary_response(response: object) -> None:
    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)  # type: ignore[arg-type]


@pytest.mark.parametrize("output", [None, [], "output", 1])
def test_rejects_non_dictionary_output(output: object) -> None:
    response = valid_response()
    response["output"] = output

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


@pytest.mark.parametrize("message", [None, [], "message", 1])
def test_rejects_non_dictionary_message(message: object) -> None:
    response = valid_response()
    response["output"]["message"] = message

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


@pytest.mark.parametrize("content", [None, {}, "content", 1])
def test_rejects_non_list_content(content: object) -> None:
    response = valid_response()
    response["output"]["message"]["content"] = content

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


@pytest.mark.parametrize("block", [None, [], "block", 1])
def test_rejects_non_dictionary_content_block(block: object) -> None:
    response = valid_response()
    response["output"]["message"]["content"] = [block]

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)


@pytest.mark.parametrize("text", [None, [], {}, 1, True])
def test_rejects_non_string_text(text: object) -> None:
    response = valid_response()
    response["output"]["message"]["content"] = [{"text": text}]

    with pytest.raises(ConverseResponseError):
        extract_converse_text(response)
