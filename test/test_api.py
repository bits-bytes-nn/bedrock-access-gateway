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
    monkeypatch.setattr(bedrock_module, "bedrock_model_list", {CHAT_MODEL: {"modalities": ["TEXT", "IMAGE"]}})
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


@pytest.mark.parametrize(
    "method,path", [("get", "/api/v1/models"), ("post", "/api/v1/chat/completions"), ("post", "/api/v1/embeddings")]
)
@pytest.mark.parametrize("headers,status", [({"Authorization": "Bearer wrong"}, 401), ({}, (401, 403))])
def test_every_api_route_requires_the_key(client, method, path, headers, status):
    response = getattr(client, method)(path, headers=headers)
    assert response.status_code in (status if isinstance(status, tuple) else (status,))


def test_health_check_needs_no_key(client):
    # The Lambda Web Adapter readiness check calls /health without credentials.
    assert client.get("/health").status_code == 200


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
    monkeypatch.setattr(bedrock_module, "bedrock_model_list", {arn: {"modalities": ["TEXT"]}})
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
    assert cache_points(sent_args(runtime)) == (0, 0)


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
        return {
            "contentBlockStart": {"contentBlockIndex": block, "start": {"toolUse": {"toolUseId": tool_id, "name": "f"}}}
        }

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
    indexes = [
        c["choices"][0]["delta"]["tool_calls"][0]["index"] for c in chunks if c["choices"][0]["delta"].get("tool_calls")
    ]
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
        return invoke_result(
            {"embeddings": [[0.1]] * len(texts)}, {"x-amzn-bedrock-input-token-count": str(len(texts))}
        )

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


@pytest.mark.parametrize(
    "model,expected", [("global.anthropic.claude-sonnet-5-5", "auto"), ("global.anthropic.claude-haiku-5-5", "any")]
)
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


# --- model list -----------------------------------------------------------------------


def test_missing_model_triggers_one_rate_limited_refresh(client, runtime, monkeypatch):
    calls = []

    def listing():
        calls.append(1)
        return {CHAT_MODEL: {"modalities": ["TEXT"]}, "global.anthropic.claude-opus-6": {"modalities": ["TEXT"]}}

    monkeypatch.setattr(bedrock_module, "list_bedrock_models", listing)
    monkeypatch.setattr(bedrock_module, "_model_list_refreshed_at", -1e9)
    runtime.converse.return_value = converse_result([{"text": "ok"}])

    assert chat(client, model="global.anthropic.claude-opus-6").status_code == 200
    # A second unknown model within the interval does not hit Bedrock again.
    assert chat(client, model="no-such-model").status_code == 400
    assert len(calls) == 1


def test_model_discovery_maps_profiles_to_models(monkeypatch):
    fake = bedrock_module.bedrock_client.__class__()
    profiles = {
        "SYSTEM_DEFINED": [
            {
                "inferenceProfileId": "global.anthropic.claude-opus-5-5",
                "models": [{"modelArn": "arn:aws:bedrock:::foundation-model/anthropic.claude-opus-5-5"}],
            }
        ],
        "APPLICATION": [
            {
                "inferenceProfileArn": "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile/app",
                "models": [{"modelArn": "arn:aws:bedrock:::foundation-model/anthropic.claude-opus-5-5"}],
            }
        ],
    }
    fake.get_paginator.return_value.paginate.side_effect = lambda typeEquals, **_: [
        {"inferenceProfileSummaries": profiles[typeEquals]}
    ]

    def model(model_id, status="ACTIVE", streaming=True, inference=("ON_DEMAND",)):
        return {
            "modelId": model_id,
            "responseStreamingSupported": streaming,
            "modelLifecycle": {"status": status},
            "inferenceTypesSupported": list(inference),
            "inputModalities": ["TEXT", "IMAGE"],
        }

    fake.list_foundation_models.return_value = {
        "modelSummaries": [
            model("anthropic.claude-opus-5-5", inference=("INFERENCE_PROFILE",)),
            model("amazon.nova-pro-v1:0"),
            model("anthropic.claude-v2", status="RETIRED"),
            model("cohere.rerank-v3-5:0", streaming=False),
        ]
    }
    monkeypatch.setattr(bedrock_module, "bedrock_client", fake)
    monkeypatch.setattr(bedrock_module, "profile_models", {})

    models = bedrock_module.list_bedrock_models()

    assert sorted(models) == [
        "amazon.nova-pro-v1:0",
        "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile/app",
        "global.anthropic.claude-opus-5-5",
    ]
    assert bedrock_module.profile_models["global.anthropic.claude-opus-5-5"] == "anthropic.claude-opus-5-5"


def test_openai_model_names_use_the_defaults(client, runtime, monkeypatch):
    monkeypatch.setitem(bedrock_module.bedrock_model_list, DEFAULT_MODEL, {"modalities": ["TEXT"]})
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    assert chat(client, model="gpt-4o").status_code == 200
    assert sent_args(runtime)["modelId"] == DEFAULT_MODEL

    runtime.invoke_model.side_effect = titan_response
    assert embed(client, model="text-embedding-3-small", input="a").json()["model"] == setting.DEFAULT_EMBEDDING_MODEL


def test_n_greater_than_one_is_rejected(client, runtime):
    assert chat(client, n=2).status_code == 400


# --- prompt caching ----------------------------------------------------------------

SYSTEM_AND_USER = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def cache_points(args):
    system = [block for block in args["system"] if "cachePoint" in block]
    messages = [block for message in args["messages"] for block in message["content"] if "cachePoint" in block]
    return len(system), len(messages)


@pytest.mark.parametrize(
    "model,enabled,extra_body,expected",
    [
        (CHAT_MODEL, True, None, (1, 1)),
        (CHAT_MODEL, False, None, (0, 0)),
        (CHAT_MODEL, False, {"prompt_caching": {"system": True}}, (1, 0)),
        (CHAT_MODEL, True, {"prompt_caching": {"messages": False}}, (1, 0)),
        # Claude 3 Haiku rejects cache points, so none are added even when enabled.
        ("apac.anthropic.claude-3-haiku-20240307-v1:0", True, None, (0, 0)),
    ],
)
def test_cache_points_follow_settings_and_model_support(
    client, runtime, monkeypatch, model, enabled, extra_body, expected
):
    monkeypatch.setattr(bedrock_module, "ENABLE_PROMPT_CACHING", enabled)
    monkeypatch.setitem(bedrock_module.bedrock_model_list, model, {"modalities": ["TEXT"]})
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    assert chat(client, model=model, messages=SYSTEM_AND_USER, extra_body=extra_body).status_code == 200
    assert cache_points(sent_args(runtime)) == expected


# --- message conversion ------------------------------------------------------------


def test_empty_text_blocks_are_left_out(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "again"}]},
    ]
    assert chat(client, messages=messages).status_code == 200
    args = sent_args(runtime)
    assert args["system"] == []
    assert [m["role"] for m in args["messages"]] == ["user"]
    assert args["messages"][0]["content"] == [{"text": "hi"}, {"text": "again"}]


def test_blank_user_turn_is_rejected_rather_than_dropped(client, runtime):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": [{"type": "text", "text": " "}]},
    ]
    assert chat(client, messages=messages).status_code == 400


PNG = "data:image/png;base64,iVBORw0KGgo="


def tool_turn(content):
    return [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": content},
    ]


def test_remote_images_in_tool_results_count_toward_the_cap(client, runtime, monkeypatch):
    monkeypatch.setattr(bedrock_module, "parse_image_url", lambda url: (_ for _ in ()).throw(AssertionError("fetched")))
    images = [{"type": "image_url", "image_url": {"url": f"https://example.com/{i}.png"}} for i in range(21)]
    assert chat(client, messages=tool_turn(images), tools=[TOOL]).status_code == 400


def test_malformed_tool_result_image_is_a_client_error(client, runtime):
    assert chat(client, messages=tool_turn([{"type": "image_url", "image_url": {}}]), tools=[TOOL]).status_code == 400


def test_tool_results_keep_text_and_images(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [{"type": "text", "text": '{"a": 1}'}, {"type": "image_url", "image_url": {"url": PNG}}],
        },
    ]
    assert chat(client, messages=messages, tools=[TOOL]).status_code == 200
    result = sent_args(runtime)["messages"][-1]["content"][0]["toolResult"]["content"]
    assert result[0] == {"text": '{"a": 1}'}
    assert result[1]["image"]["format"] == "png"


def test_images_in_chat_are_sent_as_image_blocks(client, runtime, monkeypatch):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    content = [{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": PNG}}]
    assert chat(client, messages=[{"role": "user", "content": content}]).status_code == 200
    assert sent_args(runtime)["messages"][0]["content"][1]["image"]["format"] == "png"

    monkeypatch.setitem(bedrock_module.bedrock_model_list, CHAT_MODEL, {"modalities": ["TEXT"]})
    assert chat(client, messages=[{"role": "user", "content": content}]).status_code == 400


# --- responses: usage, finish reasons, errors ---------------------------------------


def test_usage_counts_cached_tokens(client, runtime):
    result = converse_result([{"text": "ok"}])
    result["usage"] = {"inputTokens": 3, "outputTokens": 2, "totalTokens": 25, "cacheReadInputTokens": 20}
    runtime.converse.return_value = result
    usage = chat(client).json()["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["prompt_tokens_details"]["cached_tokens"]) == (
        23,
        2,
        20,
    )


@pytest.mark.parametrize("stream_options,expect_usage", [({"include_usage": True}, True), ({}, False), (None, False)])
def test_stream_usage_chunk_only_when_requested(client, runtime, stream_options, expect_usage):
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}}},
    ]
    runtime.converse_stream.return_value = {"stream": iter(events)}
    body = {"stream": True} if stream_options is None else {"stream": True, "stream_options": stream_options}
    lines = [line for line in chat(client, **body).text.splitlines() if line.startswith("data: {")]
    usage_chunks = [json.loads(line[6:]) for line in lines if json.loads(line[6:])["choices"] == []]
    assert bool(usage_chunks) is expect_usage
    content_chunks = [json.loads(line[6:]) for line in lines if json.loads(line[6:])["choices"]]
    assert all("usage" in chunk and chunk["usage"] is None for chunk in content_chunks)


@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("guardrail_intervened", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("malformed_tool_use", "stop"),
        ("something_new", "stop"),
    ],
)
def test_finish_reasons_stay_within_the_openai_set(client, runtime, stop_reason, expected):
    runtime.converse.return_value = converse_result([{"text": "ok"}], stop_reason=stop_reason)
    assert chat(client).json()["choices"][0]["finish_reason"] == expected


@pytest.mark.parametrize(
    "code,status,exposed",
    [
        ("ValidationException", 400, True),
        ("ResourceNotFoundException", 404, True),
        ("ThrottlingException", 429, True),
        ("ServiceQuotaExceededException", 429, True),
        ("AccessDeniedException", 403, False),
        ("ModelNotReadyException", 503, False),
        ("ServiceUnavailableException", 503, False),
        ("ModelTimeoutException", 504, False),
        ("ModelErrorException", 502, False),
        ("InternalServerException", 500, False),
    ],
)
def test_bedrock_error_codes_map_to_statuses(client, runtime, code, status, exposed):
    runtime.converse.side_effect = client_error(code, "detail-from-bedrock")
    response = chat(client)
    assert response.status_code == status
    assert ("detail-from-bedrock" in response.text) is exposed


def test_unexpected_errors_are_a_generic_500(client, runtime):
    runtime.converse.side_effect = RuntimeError("internal detail")
    response = chat(client)
    assert response.status_code == 500
    assert "internal detail" not in response.text


# --- embeddings: encoding and Nova -------------------------------------------------


def test_base64_embeddings_decode_to_float32(client, runtime):
    import base64
    import struct

    runtime.invoke_model.side_effect = titan_response
    item = embed(client, model=TITAN_V2, input="a", encoding_format="base64").json()["data"][0]
    assert struct.unpack("<2f", base64.b64decode(item["embedding"])) == pytest.approx((0.1, 0.2))


def test_nova_embeddings_use_the_requested_dimension(client, runtime):
    def nova_response(body, **_kwargs):
        dimension = json.loads(body)["singleEmbeddingParams"]["embeddingDimension"]
        return invoke_result({"embeddings": [{"embeddingType": "TEXT", "embedding": [0.0] * dimension}]})

    runtime.invoke_model.side_effect = nova_response
    model = "amazon.nova-2-multimodal-embeddings-v1:0"
    assert len(embed(client, model=model, input="a", dimensions=256).json()["data"][0]["embedding"]) == 256
    assert embed(client, model=model, input="a", dimensions=300).status_code == 400


def test_unknown_embedding_model_is_rejected(client, runtime):
    assert embed(client, model="no-such-embedder", input="a").status_code == 400


# --- OpenAI SDK contract -------------------------------------------------------------


def test_responses_parse_as_openai_sdk_models(client, runtime):
    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    runtime.converse.return_value = converse_result(
        [{"text": "ok"}, {"toolUse": {"toolUseId": "t1", "name": "get_time", "input": {}}}], stop_reason="tool_use"
    )
    ChatCompletion.model_validate(chat(client, tools=[TOOL]).json())

    for chunk in stream_chunks(
        client,
        runtime,
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ],
    ):
        ChatCompletionChunk.model_validate(chunk)

    runtime.invoke_model.side_effect = titan_response
    CreateEmbeddingResponse.model_validate(embed(client, model=TITAN_V2, input=["a", "b"]).json())


# --- auth --------------------------------------------------------------------------


def test_api_key_is_read_from_secrets_manager(monkeypatch):
    import importlib

    import boto3

    from api import auth

    secrets = boto3.client("secretsmanager")
    secrets.get_secret_value.return_value = {"SecretString": json.dumps({"api_key": "from-secret"})}
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: secrets)
    monkeypatch.setenv("API_KEY_SECRET_ARN", "arn:aws:secretsmanager:us-west-2:123456789012:secret:key")
    try:
        assert importlib.reload(auth).api_key == "from-secret"

        secrets.get_secret_value.return_value = {"SecretString": "not json"}
        with pytest.raises(RuntimeError, match="api_key"):
            importlib.reload(auth)
    finally:
        monkeypatch.delenv("API_KEY_SECRET_ARN")
        importlib.reload(auth)


# --- response_format -----------------------------------------------------------------

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
JSON_SCHEMA_FORMAT = {"type": "json_schema", "json_schema": {"name": "city", "schema": CITY_SCHEMA, "strict": True}}


def test_json_schema_response_format_becomes_output_config(client, runtime):
    runtime.converse.return_value = converse_result([{"text": '{"city": "Seoul"}'}])
    assert chat(client, response_format=JSON_SCHEMA_FORMAT).status_code == 200
    json_schema = sent_args(runtime)["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
    assert json_schema["name"] == "city"
    assert json.loads(json_schema["schema"]) == CITY_SCHEMA


@pytest.mark.parametrize("response_format", [{"type": "json_schema"}, {"type": "yaml"}])
def test_invalid_response_format_is_a_client_error(client, runtime, response_format):
    assert chat(client, response_format=response_format).status_code == 400


def test_json_object_response_format_adds_an_instruction(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "{}"}])
    chat(client, response_format={"type": "json_object"})
    args = sent_args(runtime)
    assert "outputConfig" not in args
    assert args["system"] == [{"text": bedrock_module.JSON_OBJECT_INSTRUCTION}]


def test_text_response_format_changes_nothing(client, runtime):
    runtime.converse.return_value = converse_result([{"text": "ok"}])
    chat(client, response_format={"type": "text"})
    args = sent_args(runtime)
    assert "outputConfig" not in args and args["system"] == []


def test_json_response_keeps_reasoning_out_of_the_content(client, runtime):
    runtime.converse.return_value = converse_result(
        [{"reasoningContent": {"reasoningText": {"text": "plan"}}}, {"text": '{"city": "Seoul"}'}]
    )
    body = chat(client, response_format=JSON_SCHEMA_FORMAT).json()
    assert json.loads(body["choices"][0]["message"]["content"]) == {"city": "Seoul"}
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0


def test_json_stream_keeps_reasoning_out_of_the_content(client, runtime):
    def delta(value):
        return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": value}}

    runtime.converse_stream.return_value = {
        "stream": iter(
            [
                {"messageStart": {"role": "assistant"}},
                delta({"reasoningContent": {"text": "plan"}}),
                delta({"reasoningContent": {"signature": "sig"}}),
                delta({"text": '{"city": "Seoul"}'}),
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    }
    lines = [line for line in chat(client, stream=True, response_format=JSON_SCHEMA_FORMAT).text.splitlines()]
    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    content = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks if chunk["choices"])
    assert json.loads(content) == {"city": "Seoul"}
