import base64
import json
import logging
import re
import struct
import threading
import time
from abc import ABC
from collections.abc import AsyncIterable
from typing import Literal

import boto3
import tiktoken
from botocore.config import Config
from botocore.exceptions import ClientError, ParamValidationError
from fastapi import HTTPException
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api.image_url import parse_image_url
from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.models.model_rules import ModelRules, Sampling, Thinking, rules_for
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    CompletionTokensDetails,
    Embedding,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Error,
    ErrorMessage,
    Function,
    ImageContent,
    PromptTokensDetails,
    ResponseFunction,
    TextContent,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from api.setting import (
    AWS_REGION,
    DEBUG,
    DEFAULT_MODEL,
    ENABLE_APPLICATION_INFERENCE_PROFILES,
    ENABLE_CROSS_REGION_INFERENCE,
    ENABLE_PROMPT_CACHING,
)

logger = logging.getLogger(__name__)

config = Config(
    connect_timeout=60,
    # Below the 600 second Lambda and API Gateway limits, so a stalled stream fails with an
    # error event. A stalled non-streaming call is retried and still hits the Lambda limit.
    read_timeout=550,
    retries={"max_attempts": 3, "mode": "standard"},
    max_pool_connections=50,
)

# Bedrock error codes mapped to the status returned to the caller. The message is
# passed through only for errors caused by the request itself; others could expose
# account details (for example the role ARN in an AccessDenied message).
_CLIENT_ERROR_STATUS = {
    "ValidationException": (400, True),
    "ResourceNotFoundException": (404, True),
    "ThrottlingException": (429, True),
    "ServiceQuotaExceededException": (429, True),
    "AccessDeniedException": (403, False),
    "ModelNotReadyException": (503, False),
    "ServiceUnavailableException": (503, False),
    "ModelTimeoutException": (504, False),
    "ModelErrorException": (502, False),
}


def bedrock_error_to_http(error: Exception, model_id: str) -> HTTPException:
    """Translate an exception raised by a Bedrock runtime call into an HTTP error."""
    if isinstance(error, HTTPException):
        return error
    if isinstance(error, ParamValidationError):
        # botocore rejected the request before sending it, e.g. an empty tool name.
        logger.warning("Invalid Bedrock request for model %s: %s", model_id, error)
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        # Errors raised mid-stream use camelCase codes such as "throttlingException".
        code = code[:1].upper() + code[1:]
        status, expose = _CLIENT_ERROR_STATUS.get(code, (500, False))
        log = logger.warning if status < 500 or status == 503 else logger.error
        log("Bedrock %s for model %s: %s", code or "error", model_id, error)
        detail = str(error) if expose else f"Amazon Bedrock request failed ({code or 'UnknownError'})"
        return HTTPException(status_code=status, detail=detail)
    logger.error("Bedrock invocation failed for model %s", model_id, exc_info=error)
    return HTTPException(status_code=500, detail="Internal error while calling Amazon Bedrock")


bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=AWS_REGION,
    config=config,
)

ENCODER = tiktoken.get_encoding("cl100k_base")

# Inference profile id or ARN (system defined and application) -> foundation model id,
# so model rules apply to profiles without parsing their ids.
profile_models: dict[str, str] = {}

# Bedrock rejects Claude thinking budgets below this value
MIN_THINKING_BUDGET_TOKENS = 1024

# A leading reasoning block, in the format the gateway returns reasoning in.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def _dict_image_url(part: dict) -> str | None:
    """The url of an image_url content part given as a plain dict (tool results)."""
    image_url = part.get("image_url")
    return image_url.get("url") if isinstance(image_url, dict) else None


# Remote image urls fetched for a single request. Each fetch has its own deadline,
# so this bounds how long one request can spend downloading. Converse accepts at
# most 20 images per request for most models anyway.
MAX_REMOTE_IMAGES_PER_REQUEST = 20


def list_bedrock_models() -> dict:
    """List the text models the gateway serves.

    Combines on-demand foundation models with the system defined (cross-Region) and
    application inference profiles for them, each when enabled by its setting.
    """
    global profile_models
    model_list = {}
    profiles = {}
    try:
        profile_types = []
        if ENABLE_CROSS_REGION_INFERENCE:
            profile_types.append(("SYSTEM_DEFINED", "inferenceProfileId"))
        if ENABLE_APPLICATION_INFERENCE_PROFILES:
            profile_types.append(("APPLICATION", "inferenceProfileArn"))
        paginator = bedrock_client.get_paginator("list_inference_profiles")
        for profile_type, id_key in profile_types:
            for page in paginator.paginate(maxResults=1000, typeEquals=profile_type):
                for profile in page["inferenceProfileSummaries"]:
                    models = profile.get("models") or [{}]
                    # Every model in a profile is the same model in a different Region.
                    model_arn = models[0].get("modelArn", "")
                    if profile.get(id_key) and model_arn:
                        profiles[profile[id_key]] = model_arn.split("/")[-1]

        # Only text output models; filters out image generation and similar models.
        response = bedrock_client.list_foundation_models(byOutputModality="TEXT")
        for model in response["modelSummaries"]:
            model_id = model["modelId"]
            # Streaming and lifecycle filter out rerank and retired models.
            if not model.get("responseStreamingSupported", True):
                continue
            if model.get("modelLifecycle", {}).get("status", "ACTIVE") not in ("ACTIVE", "LEGACY"):
                continue
            entry = {"modalities": model["inputModalities"]}
            if "ON_DEMAND" in model.get("inferenceTypesSupported", []):
                model_list[model_id] = entry
            for profile_id, profile_model in profiles.items():
                if profile_model == model_id:
                    model_list[profile_id] = entry
    except Exception as e:
        logger.error("Unable to list models: %s", e)
        # Keep serving the last good list rather than dropping every model but the default.
        if bedrock_model_list:
            return bedrock_model_list
    else:
        # Swap in the new mapping in one step so readers never see a partial one.
        profile_models = profiles

    if not model_list:
        # Listing failed on the first attempt; serve the default model until a refresh succeeds.
        model_list[DEFAULT_MODEL] = {"modalities": ["TEXT", "IMAGE"]}

    return model_list


# Initialize the model list.
bedrock_model_list: dict = {}
bedrock_model_list = list_bedrock_models()
_model_list_refreshed_at = time.monotonic()
_model_list_lock = threading.Lock()
# A model missing from the list triggers a refresh at most this often, so a model or
# profile enabled after cold start (or a failed first listing) recovers on its own.
MODEL_LIST_REFRESH_INTERVAL_SECONDS = 60


def refresh_model_list(force: bool = False) -> bool:
    """Reload the model list, at most once per interval unless forced. Returns whether it ran."""
    global bedrock_model_list, _model_list_refreshed_at
    with _model_list_lock:
        if not force and time.monotonic() - _model_list_refreshed_at < MODEL_LIST_REFRESH_INTERVAL_SECONDS:
            return False
        bedrock_model_list = list_bedrock_models()
        _model_list_refreshed_at = time.monotonic()
        return True


def is_model_listed(model_id: str) -> bool:
    """Whether a model is listed, refreshing the list (rate limited) when it is not."""
    if model_id in bedrock_model_list:
        return True
    if refresh_model_list():
        logger.info("Refreshed the model list after a request for %s", model_id)
    return model_id in bedrock_model_list


class BedrockModel(BaseChatModel):
    def __init__(self):
        # Per-stream state; a new BedrockModel is created for every request.
        self.think_emitted = False
        self.tool_call_indexes: dict[int, int] = {}
        self.reasoning_tokens = 0

    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        refresh_model_list(force=True)
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests. Blocking: may refresh the model list."""
        if chat_request.n is not None and chat_request.n > 1:
            raise HTTPException(status_code=400, detail="n greater than 1 is not supported")
        if not is_model_listed(chat_request.model):
            logger.warning("Unsupported model: %s", chat_request.model)
            if "application-inference-profile" in chat_request.model:
                detail = (
                    f"Application profile {chat_request.model} not found. "
                    "Available profiles can be listed via GET /models API. "
                    "Ensure ENABLE_APPLICATION_INFERENCE_PROFILES=true and "
                    "the profile exists in your AWS account."
                )
            else:
                detail = (
                    f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"
                )
            raise HTTPException(status_code=400, detail=detail)

    @staticmethod
    def _rules(model_id: str) -> ModelRules:
        """Request rules for a model id, resolving inference profiles to their model first."""
        return rules_for(profile_models.get(model_id, model_id))

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Convert the request and call Converse, mapping Bedrock errors to HTTP errors."""
        if DEBUG:
            logger.info("Raw request: %s", chat_request.model_dump_json())

        # Converting the request can fetch remote images, so it runs in the thread pool.
        try:
            args = await run_in_threadpool(self._parse_request, chat_request)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Tool call arguments must be valid JSON: {e}") from e
        if DEBUG:
            logger.info("Bedrock request: %s", args)

        try:
            if stream:
                return await run_in_threadpool(bedrock_runtime.converse_stream, **args)
            return await run_in_threadpool(bedrock_runtime.converse, **args)
        except Exception as e:
            raise bedrock_error_to_http(e, chat_request.model) from e

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Run a non-streaming chat completion."""
        response = await self._invoke_bedrock(chat_request)
        chat_response = self._create_response(
            model=chat_request.model,
            message_id=self.generate_message_id(),
            content=response["output"]["message"]["content"],
            finish_reason=response["stopReason"],
            usage=self._usage(response["usage"]),
        )
        if DEBUG:
            logger.info("Proxy response: %s", chat_response.model_dump_json())
        return chat_response

    @staticmethod
    def _usage(usage: dict, reasoning_tokens: int = 0) -> Usage:
        """Map Converse token usage to OpenAI usage."""
        # totalTokens also counts cache reads and writes, which OpenAI reports as prompt tokens.
        prompt_tokens = usage["totalTokens"] - usage["outputTokens"]
        cache_read_tokens = usage.get("cacheReadInputTokens", 0)
        cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=usage["outputTokens"],
            total_tokens=usage["totalTokens"],
            prompt_tokens_details=(
                PromptTokensDetails(cached_tokens=cache_read_tokens, audio_tokens=0)
                if cache_read_tokens or cache_write_tokens
                else None
            ),
            # Bedrock does not report reasoning tokens; the gateway estimates them.
            completion_tokens_details=(
                CompletionTokensDetails(reasoning_tokens=reasoning_tokens, audio_tokens=0) if reasoning_tokens else None
            ),
        )

    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Start a streaming chat completion.

        Bedrock is called before the first event is produced, so request errors are
        raised here and returned with a proper status code instead of inside a 200 stream.
        """
        response = await self._invoke_bedrock(chat_request, stream=True)
        return self._stream_events(chat_request, response["stream"])

    async def _stream_events(self, chat_request: ChatRequest, stream) -> AsyncIterable[bytes]:
        """Translate a Bedrock event stream into OpenAI server-sent events."""
        message_id = self.generate_message_id()
        try:
            # Reading the next event blocks on the socket, so do it in the thread pool.
            async for chunk in iterate_in_threadpool(stream):
                stream_response = self._create_response_stream(
                    model_id=chat_request.model, message_id=message_id, chunk=chunk
                )
                if not stream_response:
                    continue
                if DEBUG:
                    logger.info("Proxy response: %s", stream_response.model_dump_json())
                if stream_response.choices:
                    yield self.stream_response_to_bytes(stream_response)
                elif chat_request.stream_options and chat_request.stream_options.include_usage:
                    # An empty choices for Usage as per OpenAI doc below:
                    # if you set stream_options: {"include_usage": true}.
                    # an additional chunk will be streamed before the data: [DONE] message.
                    # The usage field on this chunk shows the token usage statistics for the entire request,
                    # and the choices field will always be an empty array.
                    # All other chunks will also include a usage field, but with a null value.
                    yield self.stream_response_to_bytes(stream_response)

            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()
        except Exception as e:
            # Headers are already sent, so the error can only be reported as an event.
            error = bedrock_error_to_http(e, chat_request.model)
            yield self.stream_response_to_bytes(Error(error=ErrorMessage(message=str(error.detail))))
        finally:
            # Also runs when the client disconnects, so the Bedrock connection is not left open.
            close = getattr(stream, "close", None)
            if close:
                close()

    def _cache_enabled(self, chat_request: ChatRequest, key: str) -> bool:
        """Whether to add a cache point to ``key`` ("system" or "messages").

        extra_body.prompt_caching.<key> overrides ENABLE_PROMPT_CACHING, and models
        without prompt caching never get one (Bedrock rejects the request otherwise).
        See: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        """
        if not self._rules(chat_request.model).prompt_caching:
            return False
        prompt_caching = (chat_request.extra_body or {}).get("prompt_caching")
        if isinstance(prompt_caching, dict) and key in prompt_caching:
            return prompt_caching[key] is True
        return ENABLE_PROMPT_CACHING

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict]:
        """Collect system and developer messages as Converse system blocks."""
        system_prompts = []
        for message in chat_request.messages:
            if message.role not in ("system", "developer"):
                continue
            if isinstance(message.content, str):
                text = message.content
            else:
                text = "\n".join(part.text for part in message.content)
            # Converse rejects an empty system text block.
            if text.strip():
                system_prompts.append({"text": text})

        if system_prompts and self._cache_enabled(chat_request, "system"):
            system_prompts.append({"cachePoint": {"type": "default"}})
        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """Convert user, assistant and tool messages to Converse messages.

        See: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append({"role": "user", "content": self._parse_content_parts(message, chat_request.model)})
            elif isinstance(message, AssistantMessage):
                # Converse rejects blank text blocks, so empty assistant text is left out.
                content = self._parse_content_parts(message, chat_request.model) if message.content else []
                for tool_call in message.tool_calls or []:
                    arguments = tool_call.function.arguments
                    tool_input = json.loads(arguments) if arguments and arguments.strip() else {}
                    content.append(
                        {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.function.name, "input": tool_input}}
                    )
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif isinstance(message, ToolMessage):
                # Converse has no tool role; results go back as toolResult blocks in a user turn.
                tool_content = self._tool_result_content(message.content, chat_request.model)
                messages.append(
                    {
                        "role": "user",
                        "content": [{"toolResult": {"toolUseId": message.tool_call_id, "content": tool_content}}],
                    }
                )
        return self._merge_consecutive_roles(messages, chat_request)

    @staticmethod
    def _image_urls(chat_request: ChatRequest):
        """Yield the url of every image part, including images inside tool results."""
        for message in chat_request.messages:
            if not isinstance(message.content, list):
                continue
            for part in message.content:
                if isinstance(part, ImageContent):
                    yield part.image_url.url
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    yield _dict_image_url(part)

    def _image_block(self, url: str, model_id: str) -> dict:
        """Fetch or decode an image url into a Converse image block."""
        if not self.is_supported_modality(model_id, modality="IMAGE"):
            raise HTTPException(status_code=400, detail=f"Multimodal message is currently not supported by {model_id}")
        image_data, content_type = parse_image_url(url)
        return {"image": {"format": content_type.removeprefix("image/"), "source": {"bytes": image_data}}}

    def _tool_result_content(self, content: str | list[dict], model_id: str) -> list[dict]:
        """Convert OpenAI tool message content into Converse tool result blocks.

        Handles a plain string, a list of text parts (OpenAI SDK 1.91.0+) and image
        parts, which Converse accepts in tool results. Text is passed through as is.
        """
        if isinstance(content, str):
            return [{"text": content}]

        blocks = []
        for item in content:
            if item.get("type") == "image_url":
                url = _dict_image_url(item)
                if not isinstance(url, str):
                    raise HTTPException(status_code=400, detail="Tool result image_url must contain a url")
                blocks.append(self._image_block(url, model_id))
            elif isinstance(item.get("text"), str):
                blocks.append({"text": item["text"]})
            else:
                blocks.append({"text": json.dumps(item)})
        return blocks or [{"text": ""}]

    def _merge_consecutive_roles(self, messages: list[dict], chat_request: ChatRequest) -> list[dict]:
        """Merge consecutive messages of the same role, which Converse requires to alternate.

        Also makes the conversation end with a user turn for models that reject
        assistant prefill, and adds the message cache point.
        """
        merged = []
        for message in messages:
            if merged and merged[-1]["role"] == message["role"]:
                merged[-1]["content"].extend(message["content"])
            else:
                merged.append({"role": message["role"], "content": list(message["content"])})

        if merged and merged[-1]["role"] == "assistant" and not self._rules(chat_request.model).prefill:
            merged.append(
                {"role": "user", "content": [{"text": "Please continue your response from where you left off."}]}
            )

        if merged and self._cache_enabled(chat_request, "messages"):
            last_user = next((message for message in reversed(merged) if message["role"] == "user"), None)
            if last_user:
                last_user["content"].append({"cachePoint": {"type": "default"}})
        return merged

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create the Converse request for an OpenAI chat request.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        remote_images = sum(
            1 for url in self._image_urls(chat_request) if not (isinstance(url, str) and url.startswith("data:"))
        )
        if remote_images > MAX_REMOTE_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"A request may reference at most {MAX_REMOTE_IMAGES_PER_REQUEST} remote image urls",
            )
        rules = self._rules(chat_request.model)
        messages = self._parse_messages(chat_request)
        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": self._parse_system_prompts(chat_request),
        }

        max_tokens = (
            chat_request.max_completion_tokens
            if chat_request.max_completion_tokens is not None
            else chat_request.max_tokens
        )
        inference_config = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if chat_request.temperature is not None:
            inference_config["temperature"] = chat_request.temperature
        if chat_request.top_p is not None:
            inference_config["topP"] = chat_request.top_p
        if chat_request.stop is not None:
            inference_config["stopSequences"] = (
                [chat_request.stop] if isinstance(chat_request.stop, str) else chat_request.stop
            )

        if chat_request.response_format and chat_request.response_format.get("type") not in (None, "text"):
            logger.warning(
                "response_format %s is not supported and is ignored for %s",
                chat_request.response_format.get("type"),
                chat_request.model,
            )

        additional_fields = self._reasoning_fields(chat_request, rules, max_tokens)
        if chat_request.extra_body:
            # Passed through to the model, except prompt_caching which is the gateway's own field.
            additional_fields.update({k: v for k, v in chat_request.extra_body.items() if k != "prompt_caching"})
        if additional_fields:
            args["additionalModelRequestFields"] = additional_fields
        budget_thinking = rules.thinking is Thinking.BUDGET and any(
            isinstance(additional_fields.get(key), dict) and additional_fields[key].get("type") != "disabled"
            for key in ("reasoning_config", "thinking")
        )

        if rules.sampling is Sampling.NONE or budget_thinking:
            inference_config.pop("temperature", None)
            inference_config.pop("topP", None)
        elif rules.sampling is Sampling.ONE and "temperature" in inference_config:
            # Both together are rejected; temperature takes precedence.
            inference_config.pop("topP", None)
        args["inferenceConfig"] = inference_config

        # Converse has no "none" tool choice. Leaving tools out is the closest match, but
        # Bedrock requires the tool config whenever the history contains tool blocks.
        skip_tools = chat_request.tool_choice == "none" and not self._has_tool_blocks(messages)
        if chat_request.tools and not skip_tools:
            tool_config = {"tools": [self._convert_tool_spec(t.function) for t in chat_request.tools]}
            # Some models reject toolChoice; "none" cannot be expressed, so it falls back to auto.
            if chat_request.tool_choice and chat_request.tool_choice != "none" and rules.tool_choice:
                tool_choice = self._convert_tool_choice(chat_request.tool_choice)
                if "auto" not in tool_choice and (budget_thinking or not rules.forced_tool_choice):
                    logger.warning("%s does not accept a forced tool choice here, using auto", chat_request.model)
                    tool_choice = {"auto": {}}
                tool_config["toolChoice"] = tool_choice
            args["toolConfig"] = tool_config
        return args

    def _reasoning_fields(self, chat_request: ChatRequest, rules: ModelRules, max_tokens: int | None) -> dict:
        """Translate reasoning_effort into the model's additionalModelRequestFields."""
        effort = chat_request.reasoning_effort
        if not effort:
            return {}
        if rules.thinking is Thinking.ADAPTIVE:
            return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
        if rules.thinking is Thinking.EFFORT_STRING:
            return {"reasoning_config": effort}
        if rules.thinking is Thinking.BUDGET:
            if max_tokens is None:
                raise HTTPException(
                    status_code=400,
                    detail="max_tokens or max_completion_tokens must be set when using reasoning_effort with this model",
                )
            budget_tokens = max(MIN_THINKING_BUDGET_TOKENS, self._calc_budget_tokens(max_tokens, effort))
            if budget_tokens >= max_tokens:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tokens must be greater than {MIN_THINKING_BUDGET_TOKENS} "
                    "when using reasoning_effort with this model",
                )
            return {"reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}}
        logger.debug("reasoning_effort is ignored for %s", chat_request.model)
        return {}

    @staticmethod
    def _convert_tool_choice(tool_choice: str | dict) -> dict:
        """Map an OpenAI tool_choice ("auto", "required" or a named function) to Converse."""
        if tool_choice == "required":
            return {"any": {}}
        if isinstance(tool_choice, str):
            return {"auto": {}}
        if not isinstance(tool_choice, dict) or "function" not in tool_choice:
            raise HTTPException(
                status_code=400,
                detail="tool_choice must contain 'function' key when specifying a specific tool",
            )
        return {"tool": {"name": tool_choice["function"].get("name", "")}}

    @staticmethod
    def _has_tool_blocks(messages: list[dict]) -> bool:
        return any("toolUse" in block or "toolResult" in block for message in messages for block in message["content"])

    def _create_response(
        self,
        model: str,
        message_id: str,
        content: list[dict],
        finish_reason: str | None,
        usage: Usage,
    ) -> ChatResponse:
        # A response can hold several text blocks, reasoning, and tool calls together.
        text_parts = []
        reasoning_parts = []
        tool_calls = []
        for part in content:
            if "text" in part:
                text_parts.append(part["text"])
            elif "reasoningContent" in part:
                # redactedContent is encrypted and has no text to surface
                reasoning_parts.append(part["reasoningContent"].get("reasoningText", {}).get("text", ""))
            elif "toolUse" in part:
                tool = part["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tool["toolUseId"],
                        type="function",
                        function=ResponseFunction(name=tool["name"], arguments=json.dumps(tool["input"])),
                    )
                )
            else:
                logger.warning("Unknown content block in response: %s", ",".join(part.keys()))

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        if reasoning:
            text = f"<think>{reasoning}</think>{text}"
            usage.completion_tokens_details = CompletionTokensDetails(
                reasoning_tokens=len(ENCODER.encode(reasoning)), audio_tokens=0
            )
        message = ChatResponseMessage(role="assistant", content=(text or None) if tool_calls else text)
        if tool_calls:
            message.tool_calls = tool_calls

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=usage,
        )
        # The route excludes unset fields, so the defaults are set explicitly.
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(self, model_id: str, message_id: str, chunk: dict) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        if DEBUG:
            logger.info("Bedrock response chunk: %s", chunk)

        finish_reason = None
        message = None

        if "messageStart" in chunk:
            message = ChatResponseMessage(
                role=chunk["messageStart"]["role"],
                content="",
            )

        if "contentBlockStart" in chunk:
            # tool call start
            delta = chunk["contentBlockStart"]["start"]
            if "toolUse" in delta:
                # OpenAI tool call indexes count tool calls only, starting at 0.
                block_index = chunk["contentBlockStart"]["contentBlockIndex"]
                index = self.tool_call_indexes.setdefault(block_index, len(self.tool_call_indexes))
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            type="function",
                            id=delta["toolUse"]["toolUseId"],
                            function=ResponseFunction(
                                name=delta["toolUse"]["name"],
                                arguments="",
                            ),
                        )
                    ]
                )
                # Close an open reasoning block before the tool call starts.
                closing_tag = self._close_think_tag()
                if closing_tag:
                    message.content = closing_tag

        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                # Regular text content - close thinking tag if open
                content = delta["text"]
                if self.think_emitted:
                    # Transition from reasoning to regular text
                    content = "</think>" + content
                    self.think_emitted = False
                message = ChatResponseMessage(content=content)
            elif "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    content = delta["reasoningContent"]["text"]
                    self.reasoning_tokens += len(ENCODER.encode(content))
                    if not self.think_emitted:
                        # Start of reasoning content
                        content = "<think>" + content
                        self.think_emitted = True
                    message = ChatResponseMessage(content=content)
                elif "signature" in delta["reasoningContent"]:
                    # Port of "signature_delta" (for models that send it)
                    if self.think_emitted:
                        message = ChatResponseMessage(content="</think>")
                        self.think_emitted = False
                    else:
                        return None  # Ignore signature if no <think> started
                else:
                    return None  # e.g. encrypted redactedContent, nothing to surface
            elif "toolUse" in delta:
                block_index = chunk["contentBlockDelta"]["contentBlockIndex"]
                index = self.tool_call_indexes.setdefault(block_index, len(self.tool_call_indexes))
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            function=ResponseFunction(
                                arguments=delta["toolUse"]["input"],
                            ),
                        )
                    ]
                )

        if "messageStop" in chunk:
            # Close any open reasoning block in the same chunk that carries finish_reason.
            closing_tag = self._close_think_tag()
            message = ChatResponseMessage(content=closing_tag) if closing_tag else ChatResponseMessage()
            finish_reason = chunk["messageStop"]["stopReason"]

        if "metadata" in chunk and "usage" in chunk["metadata"]:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[],
                usage=self._usage(chunk["metadata"]["usage"], self.reasoning_tokens),
            )

        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                # OpenAI sends "usage": null on every chunk but the final usage chunk.
                usage=None,
            )

        return None

    def _close_think_tag(self) -> str | None:
        """Return the closing reasoning tag if a reasoning block is open, and mark it closed."""
        if not self.think_emitted:
            return None
        self.think_emitted = False
        return "</think>"

    def _parse_content_parts(self, message: UserMessage | AssistantMessage, model_id: str) -> list[dict]:
        parts = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
        content_parts = []
        for part in parts:
            if isinstance(part, TextContent):
                text = part.text
                if isinstance(message, AssistantMessage):
                    # Reasoning the gateway returned earlier is not sent back as answer text.
                    text = _THINK_BLOCK.sub("", text, count=1)
                # Converse rejects blank text blocks.
                if text.strip():
                    content_parts.append({"text": text})
            elif isinstance(part, ImageContent):
                content_parts.append(self._image_block(part.image_url.url, model_id))
        if not content_parts and isinstance(message, UserMessage):
            # Dropping the whole turn would silently change how the conversation ends.
            raise HTTPException(status_code=400, detail="User message content must not be empty")
        return content_parts

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        return modality in bedrock_model_list.get(model_id, {}).get("modalities", [])

    def _convert_tool_spec(self, func: Function) -> dict:
        tool_spec = {
            "name": func.name,
            # OpenAI allows omitting parameters for a function that takes none.
            "inputSchema": {"json": func.parameters or {"type": "object", "properties": {}}},
        }
        # description is optional in OpenAI, but Converse rejects a null value.
        if func.description:
            tool_spec["description"] = func.description
        return {"toolSpec": tool_spec}

    def _calc_budget_tokens(self, max_tokens: int, reasoning_effort: Literal["low", "medium", "high"]) -> int:
        # Helper function to calculate budget_tokens based on the max_tokens.
        # Ratio for efforts: low - 30%, medium - 60%, high - 80%. High leaves room for
        # the answer; spending almost all of max_tokens on thinking returns empty content.
        # The caller raises the result to the 1,024 token minimum.
        if reasoning_effort == "low":
            return int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            return int(max_tokens * 0.6)
        else:
            return int(max_tokens * 0.8)

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "content_filtered": "content_filter",
                "guardrail_intervened": "content_filter",
                "model_context_window_exceeded": "length",
                # Not in the OpenAI set; the closest match is a normal stop.
                "malformed_model_output": "stop",
                "malformed_tool_use": "stop",
            }
            reason = finish_reason.lower()
            if reason not in finish_reason_mapping:
                # Strict OpenAI clients reject values outside their enum.
                logger.warning("Unknown Bedrock stop reason %s reported as stop", finish_reason)
            elif reason.startswith("malformed_"):
                logger.warning("Bedrock stopped with %s", finish_reason)
            return finish_reason_mapping.get(reason, "stop")
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    # Same cap as the OpenAI embeddings API.
    MAX_INPUTS = 2048

    def _input_texts(self, embeddings_request: EmbeddingsRequest) -> list[str]:
        """Normalize the OpenAI ``input`` field to a list of texts.

        A list of ints is a single tokenized input and a list of lists is a batch of
        them; token ids are decoded with tiktoken since Bedrock only accepts text.
        """
        value = embeddings_request.input
        if isinstance(value, str):
            texts = [value]
        elif not value:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        elif isinstance(value[0], int):
            texts = [ENCODER.decode(value)]
        elif isinstance(value[0], list):
            texts = [ENCODER.decode(tokens) for tokens in value]
        else:
            texts = list(value)

        if len(texts) > self.MAX_INPUTS:
            raise HTTPException(
                status_code=400,
                detail=f"Input must not contain more than {self.MAX_INPUTS} items",
            )
        return texts

    @staticmethod
    def _reject_dimensions(embeddings_request: EmbeddingsRequest) -> None:
        if embeddings_request.dimensions is not None:
            raise HTTPException(
                status_code=400,
                detail=f"dimensions is not supported by {embeddings_request.model}",
            )

    def _invoke_model(self, args: dict, model_id: str) -> tuple[dict, dict]:
        """Call InvokeModel and return the parsed response body and the HTTP headers."""
        body = json.dumps(args)
        if DEBUG:
            logger.info("Bedrock request for %s: %s", model_id, body)
        try:
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
            # The body is streamed, so reading it can fail like the call itself.
            response_body = json.loads(response["body"].read())
        except Exception as e:
            raise bedrock_error_to_http(e, model_id) from e
        if DEBUG:
            logger.info("Bedrock response keys: %s", list(response_body))
        return response_body, response.get("ResponseMetadata", {}).get("HTTPHeaders", {})

    def _create_response(
        self,
        embeddings: list[list[float]],
        model: str,
        input_tokens: int,
        encoding_format: Literal["float", "base64"],
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                # Little-endian float32, the layout OpenAI clients decode.
                encoded_embedding = base64.b64encode(struct.pack(f"<{len(embedding)}f", *embedding))
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens,
            ),
        )
        return response


class CohereEmbeddingsModel(BedrockEmbeddingsModel):
    # Bedrock accepts at most 96 texts per Cohere Embed request.
    BATCH_SIZE = 96
    # Embed v4 output sizes; v3 has a fixed size.
    V4_DIMENSIONS = {256, 512, 1024, 1536}

    def __init__(self, is_v4: bool = False):
        self.is_v4 = is_v4

    def _parse_args(self, texts: list[str], dimensions: int | None) -> dict:
        args = {
            "texts": texts,
            "input_type": "search_document",
            # Inputs longer than the model limit are cut from the end.
            "truncate": "END",
        }
        if self.is_v4:
            args["embedding_types"] = ["float"]
            if dimensions is not None:
                args["output_dimension"] = dimensions
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        dimensions = embeddings_request.dimensions
        if not self.is_v4:
            self._reject_dimensions(embeddings_request)
        elif dimensions is not None and dimensions not in self.V4_DIMENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimensions {dimensions}. Must be one of {sorted(self.V4_DIMENSIONS)}",
            )
        texts = self._input_texts(embeddings_request)

        all_embeddings = []
        total_tokens = 0
        for start in range(0, len(texts), self.BATCH_SIZE):
            response_body, headers = self._invoke_model(
                args=self._parse_args(texts[start : start + self.BATCH_SIZE], dimensions),
                model_id=embeddings_request.model,
            )
            embeddings = response_body["embeddings"]
            # v4 (and v3 with embedding_types) returns {"float": [...]} instead of a list.
            all_embeddings.extend(embeddings["float"] if isinstance(embeddings, dict) else embeddings)
            # Cohere reports usage only in the response headers.
            total_tokens += int(headers.get("x-amzn-bedrock-input-token-count", 0))

        return self._create_response(
            embeddings=all_embeddings,
            model=embeddings_request.model,
            input_tokens=total_tokens,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):
    # Titan Text Embeddings V2 output sizes; V1 has a fixed size.
    V2_DIMENSIONS = {256, 512, 1024}

    def __init__(self, is_v2: bool = False):
        self.is_v2 = is_v2

    def _parse_args(self, input_text: str, dimensions: int | None) -> dict:
        args = {"inputText": input_text}
        if dimensions is not None:
            args["dimensions"] = dimensions
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        dimensions = embeddings_request.dimensions
        if not self.is_v2:
            self._reject_dimensions(embeddings_request)
        elif dimensions is not None and dimensions not in self.V2_DIMENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimensions {dimensions}. Must be one of {sorted(self.V2_DIMENSIONS)}",
            )

        # Titan takes a single text per InvokeModel call, while the OpenAI contract
        # allows a list, so embed the items one by one.
        all_embeddings = []
        total_tokens = 0
        for text in self._input_texts(embeddings_request):
            response_body, _ = self._invoke_model(
                args=self._parse_args(text, dimensions),
                model_id=embeddings_request.model,
            )
            all_embeddings.append(response_body["embedding"])
            total_tokens += response_body["inputTextTokenCount"]

        return self._create_response(
            embeddings=all_embeddings,
            model=embeddings_request.model,
            input_tokens=total_tokens,
            encoding_format=embeddings_request.encoding_format,
        )


class NovaEmbeddingsModel(BedrockEmbeddingsModel):
    # Per https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
    VALID_DIMENSIONS = {256, 384, 1024, 3072}
    DEFAULT_DIMENSION = 3072

    def _parse_args(self, text: str, dimension: int) -> dict:
        return {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                # Nova supports 9 embeddingPurpose values; GENERIC_INDEX is the
                # general-purpose default suitable for most retrieval use cases.
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "text": {
                    "truncationMode": "END",
                    "value": text,
                },
            },
        }

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        dimension = embeddings_request.dimensions or self.DEFAULT_DIMENSION
        if dimension not in self.VALID_DIMENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimensions {dimension}. Must be one of {sorted(self.VALID_DIMENSIONS)}",
            )

        all_embeddings = []
        total_tokens = 0
        for idx, text in enumerate(self._input_texts(embeddings_request)):
            response_body, _ = self._invoke_model(
                args=self._parse_args(text, dimension),
                model_id=embeddings_request.model,
            )

            # Response: {"embeddings": [{"embeddingType": "TEXT", "embedding": [...]}]}
            embeddings_list = response_body.get("embeddings", [])
            if not embeddings_list:
                raise HTTPException(
                    status_code=502,
                    detail=f"No embeddings returned from Nova model for input[{idx}]",
                )
            all_embeddings.append(embeddings_list[0]["embedding"])
            # Nova doesn't return token counts in the response; approximate with cl100k_base
            total_tokens += len(ENCODER.encode(text))

        return self._create_response(
            embeddings=all_embeddings,
            model=embeddings_request.model,
            input_tokens=total_tokens,
            encoding_format=embeddings_request.encoding_format,
        )


# Embedding models by foundation model id.
EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": CohereEmbeddingsModel,
    "cohere.embed-english-v3": CohereEmbeddingsModel,
    "cohere.embed-v4:0": lambda: CohereEmbeddingsModel(is_v4=True),
    "amazon.titan-embed-text-v1": TitanEmbeddingsModel,
    "amazon.titan-embed-text-v2:0": lambda: TitanEmbeddingsModel(is_v2=True),
    "amazon.nova-2-multimodal-embeddings-v1:0": NovaEmbeddingsModel,
}

# Inference profile prefixes, e.g. global.cohere.embed-v4:0 for a model only served through a profile.
_PROFILE_PREFIXES = ("global.", "us.", "eu.", "apac.", "jp.", "au.", "ca.", "us-gov.")


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    base_model_id = model_id
    for prefix in _PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            base_model_id = model_id.removeprefix(prefix)
            break
    factory = EMBEDDING_MODELS.get(base_model_id)
    if factory is None:
        logger.warning("Unsupported embedding model id %s", model_id)
        raise HTTPException(status_code=400, detail=f"Unsupported embedding model id {model_id}")
    return factory()
