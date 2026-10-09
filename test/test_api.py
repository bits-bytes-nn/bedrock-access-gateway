"""HTTP-level tests for the gateway routes, with Bedrock replaced by mocks."""

import io
import json

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from api import setting
from api.app import app
from api.models import bedrock as bedrock_module
from api.setting import DEFAULT_MODEL

AUTH = {"Authorization": "Bearer test-api-key"}
CHAT_MODEL = "global.anthropic.claude-sonnet-5-5"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def runtime(monkeypatch):
    """A fresh mock of the bedrock-runtime client for each test."""
    fake = bedrock_module.bedrock_runtime.__class__()
    monkeypatch.setattr(bedrock_module, "bedrock_runtime", fake)
    monkeypatch.setattr(
        bedrock_module, "bedrock_model_list", {CHAT_MODEL: {"modalities": ["TEXT", "IMAGE"]}}
    )
    return fake


def client_error(code, message="boom"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


def converse_result(content, stop_reason="end_turn"):
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "stopReason": stop_reason,
    }


def invoke_result(body, headers=None):
    return {"body": io.BytesIO(json.dumps(body).encode()), "ResponseMetadata": {"HTTPHeaders": headers or {}}}


def chat(client, **body):
    body.setdefault("model", CHAT_MODEL)
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post("/api/v1/chat/completions", json=body, headers=AUTH)


def embed(client, **body):
    return client.post("/api/v1/embeddings", json=body, headers=AUTH)


# --- auth and request validation -------------------------------------------------


def test_wrong_api_key_is_rejected(client):
    response = client.get("/api/v1/models", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_validation_error_is_json_without_internals(client):
    response = embed(client, model="amazon.titan-embed-text-v2:0")
    assert response.status_code == 400
    assert "input" in response.json()["detail"]
    assert "Traceback" not in response.text and "File " not in response.text


# --- models ------------------------------------------------------------------------


def test_unknown_model_is_not_found(client):
    assert client.get("/api/v1/models/no-such-model", headers=AUTH).status_code == 404
    assert client.get(f"/api/v1/models/{DEFAULT_MODEL}", headers=AUTH).status_code == 200


def test_model_lookup_accepts_arn_ids(client, monkeypatch):
    arn = "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile/abc"
    monkeypatch.setattr(bedrock_module.BedrockModel, "list_models", lambda self: [arn])
    assert client.get(f"/api/v1/models/{arn}", headers=AUTH).status_code == 200


def test_failed_refresh_keeps_the_previous_model_list(monkeypatch):
    previous = {CHAT_MODEL: {"modalities": ["TEXT"]}}
    monkeypatch.setattr(bedrock_module, "bedrock_model_list", previous)
    failing = bedrock_module.bedrock_client.__class__()
    failing.list_foundation_models.side_effect = client_error("ThrottlingException")
    monkeypatch.setattr(bedrock_module, "bedrock_client", failing)
    assert bedrock_module.list_bedrock_models() == previous


# --- chat: error mapping -----------------------------------------------------------


def test_access_denied_does_not_leak_account_details(client, runtime):
    runtime.converse.side_effect = client_error(
        "AccessDeniedException", "arn:aws:sts::123456789012:assumed-role/X is not authorized"
    )
    response = chat(client)
    assert response.status_code == 403
    assert "123456789012" not in response.text


def test_validation_error_message_is_passed_through(client, runtime):
    runtime.converse.side_effect = client_error("ValidationException", "max_tokens is too large")
    response = chat(client)
    assert response.status_code == 400
    assert "max_tokens is too large" in response.json()["detail"]


def test_stream_request_errors_return_a_status_code(client, runtime):
    runtime.converse_stream.side_effect = client_error("ValidationException", "bad request")
    response = chat(client, stream=True)
    assert response.status_code == 400


def test_mid_stream_error_is_reported_as_an_event(client, runtime):
    def events():
        yield {"messageStart": {"role": "assistant"}}
        raise client_error("throttlingException", "slow down")

    runtime.converse_stream.return_value = {"stream": events()}
    response = chat(client, stream=True)
    assert response.status_code == 200
    assert "slow down" in response.text


@pytest.mark.parametrize("arguments", ["{not json", "[1"])
def test_invalid_tool_call_arguments_are_a_client_error(client, runtime, arguments):
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": arguments}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]
    assert chat(client, messages=messages).status_code == 400


def test_prompt_caching_flag_with_wrong_shape_is_ignored(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    response = chat(
        client,
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        extra_body={"prompt_caching": True},
    )
    assert response.status_code == 200


# --- chat: request conversion ------------------------------------------------------

TOOL = {"type": "function", "function": {"name": "get_time"}}


def sent_args(runtime):
    return runtime.converse.call_args.kwargs


def test_tool_without_description_or_parameters_is_accepted(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    assert chat(client, tools=[TOOL]).status_code == 200
    spec = sent_args(runtime)["toolConfig"]["tools"][0]["toolSpec"]
    assert "description" not in spec
    assert spec["inputSchema"] == {"json": {"type": "object", "properties": {}}}


def test_tool_choice_none_leaves_tools_out(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    chat(client, tools=[TOOL], tool_choice="none")
    assert "toolConfig" not in sent_args(runtime)


def test_tool_choice_none_keeps_tools_when_history_uses_them(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    messages = [
        {"role": "user", "content": "time?"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "get_time", "arguments": ""}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "noon"},
    ]
    chat(client, messages=messages, tools=[TOOL], tool_choice="none")
    tool_config = sent_args(runtime)["toolConfig"]
    assert "toolChoice" not in tool_config


def test_system_message_may_be_a_list_of_text_parts(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": "hi"},
    ]
    assert chat(client, messages=messages).status_code == 200
    assert sent_args(runtime)["system"][0] == {"text": "a\nb"}


def test_high_budget_ratio():
    assert bedrock_module.BedrockModel()._calc_budget_tokens(10_000, "high") == 8_000


# --- chat: response conversion -----------------------------------------------------


def test_all_text_blocks_and_tool_calls_are_returned(client, runtime):
    runtime.converse.return_value = converse_result(
        [
            {"text": "Let me "},
            {"text": "check."},
            {"toolUse": {"toolUseId": "t1", "name": "get_time", "input": {}}},
        ],
        stop_reason="tool_use",
    )
    message = chat(client, tools=[TOOL]).json()["choices"][0]["message"]
    assert message["content"] == "Let me check."
    assert message["tool_calls"][0]["function"]["name"] == "get_time"


def stream_chunks(client, runtime, events):
    runtime.converse_stream.return_value = {"stream": iter(events)}
    lines = [line[6:] for line in chat(client, stream=True).text.splitlines() if line.startswith("data: ")]
    return [json.loads(line) for line in lines if line != "[DONE]"]


def test_stream_tool_call_indexes_count_tool_calls_only(client, runtime):
    def tool_start(block, tool_id):
        return {"contentBlockStart": {"contentBlockIndex": block, "start": {"toolUse": {"toolUseId": tool_id, "name": "f"}}}}

    def tool_delta(block):
        return {"contentBlockDelta": {"contentBlockIndex": block, "delta": {"toolUse": {"input": "{}"}}}}

    chunks = stream_chunks(
        client,
        runtime,
        [
            {"messageStart": {"role": "assistant"}},
            tool_start(0, "a"),
            tool_delta(0),
            tool_start(1, "b"),
            tool_delta(1),
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )
    indexes = [c["choices"][0]["delta"]["tool_calls"][0]["index"] for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert indexes == [0, 0, 1, 1]


def test_stream_keeps_finish_reason_when_reasoning_is_open(client, runtime):
    chunks = stream_chunks(
        client,
        runtime,
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}},
            {"messageStop": {"stopReason": "max_tokens"}},
        ],
    )
    last = chunks[-1]["choices"][0]
    assert last["finish_reason"] == "length"
    assert last["delta"]["content"] == "</think>"


def test_stream_closes_reasoning_before_a_tool_call(client, runtime):
    chunks = stream_chunks(
        client,
        runtime,
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "plan"}}}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "a", "name": "f"}}}},
            {"messageStop": {"stopReason": "tool_use"}},
        ],
    )
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == "<think>plan</think>"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# --- embeddings --------------------------------------------------------------------

TITAN_V2 = "amazon.titan-embed-text-v2:0"


def titan_response(*_args, **_kwargs):
    return invoke_result({"embedding": [0.1, 0.2], "inputTextTokenCount": 2})


@pytest.mark.parametrize(
    "value,expected_count",
    [("a", 1), (["a", "b"], 2), ([1, 2, 3], 1), ([[1, 2], [3]], 2)],
)
def test_titan_accepts_every_openai_input_form(client, runtime, value, expected_count):
    runtime.invoke_model.side_effect = titan_response
    response = embed(client, model=TITAN_V2, input=value)
    assert response.status_code == 200
    assert len(response.json()["data"]) == expected_count


@pytest.mark.parametrize("value", [[], [1, "a"], [["a"]]])
def test_invalid_embedding_input_is_a_client_error(client, runtime, value):
    runtime.invoke_model.side_effect = titan_response
    assert embed(client, model=TITAN_V2, input=value).status_code == 400


def test_titan_v2_passes_dimensions(client, runtime):
    runtime.invoke_model.side_effect = titan_response
    assert embed(client, model=TITAN_V2, input="a", dimensions=256).status_code == 200
    assert json.loads(runtime.invoke_model.call_args.kwargs["body"])["dimensions"] == 256
    assert embed(client, model=TITAN_V2, input="a", dimensions=300).status_code == 400
    assert embed(client, model="amazon.titan-embed-text-v1", input="a", dimensions=256).status_code == 400


def test_cohere_splits_large_batches_and_reports_usage(client, runtime):
    def cohere_response(body, **_kwargs):
        texts = json.loads(body)["texts"]
        return invoke_result({"embeddings": [[0.1]] * len(texts)}, {"x-amzn-bedrock-input-token-count": str(len(texts))})

    runtime.invoke_model.side_effect = cohere_response
    response = embed(client, model="cohere.embed-multilingual-v3", input=["t"] * 100)
    assert response.status_code == 200
    assert [len(json.loads(c.kwargs["body"])["texts"]) for c in runtime.invoke_model.call_args_list] == [96, 4]
    body = response.json()
    assert [item["index"] for item in body["data"]] == list(range(100))
    assert body["usage"]["prompt_tokens"] == 100


def test_cohere_rejects_dimensions(client, runtime):
    assert embed(client, model="cohere.embed-multilingual-v3", input="a", dimensions=256).status_code == 400


# --- settings ----------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("FALSE", False), ("0", False), ("", True)])
def test_flags_are_parsed_strictly(monkeypatch, value, expected):
    monkeypatch.setenv("SOME_FLAG", value)
    assert setting._env_flag("SOME_FLAG", True) is expected


@pytest.mark.parametrize("value", ["maybe", "ture"])
def test_unknown_flag_values_fail_fast(monkeypatch, value):
    monkeypatch.setenv("SOME_FLAG", value)
    with pytest.raises(RuntimeError):
        setting._env_flag("SOME_FLAG", False)


@pytest.mark.parametrize("value", ["10MB", "0", "-1"])
def test_invalid_size_setting_fails_fast(monkeypatch, value):
    monkeypatch.setenv("SOME_SIZE", value)
    with pytest.raises(RuntimeError):
        setting._env_positive_int("SOME_SIZE", 10)


def test_too_many_remote_images_are_rejected_before_fetching(client, runtime, monkeypatch):
    monkeypatch.setattr(bedrock_module, "parse_image_url", lambda url: (_ for _ in ()).throw(AssertionError("fetched")))
    images = [{"type": "image_url", "image_url": {"url": f"https://example.com/{i}.png"}} for i in range(21)]
    response = chat(client, messages=[{"role": "user", "content": images}])
    assert response.status_code == 400


@pytest.mark.parametrize("model,expected", [("global.anthropic.claude-sonnet-5-5", "auto"), ("global.anthropic.claude-haiku-5-5", "any")])
def test_forced_tool_choice_falls_back_where_unsupported(client, runtime, monkeypatch, model, expected):
    monkeypatch.setitem(bedrock_module.bedrock_model_list, model, {"modalities": ["TEXT"]})
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    assert chat(client, model=model, tools=[TOOL], tool_choice="required").status_code == 200
    assert list(sent_args(runtime)["toolConfig"]["toolChoice"]) == [expected]


def test_cohere_v4_through_an_inference_profile(client, runtime):
    def cohere_v4_response(body, modelId, **_kwargs):
        request = json.loads(body)
        assert modelId == "global.cohere.embed-v4:0"
        assert request["embedding_types"] == ["float"] and request["output_dimension"] == 256
        return invoke_result({"embeddings": {"float": [[0.1] * 256] * len(request["texts"])}})

    runtime.invoke_model.side_effect = cohere_v4_response
    response = embed(client, model="global.cohere.embed-v4:0", input=["a", "b"], dimensions=256)
    assert response.status_code == 200
    assert [len(item["embedding"]) for item in response.json()["data"]] == [256, 256]
