import base64
import json
import logging
import time
from abc import ABC
from typing import AsyncIterable, Literal

import boto3
import numpy as np
import tiktoken
from botocore.config import Config
from botocore.exceptions import ClientError, ParamValidationError
from fastapi import HTTPException
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api.image_url import parse_image_url
from api.models.base import BaseChatModel, BaseEmbeddingsModel
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
    # Long enough for slow streaming responses; a stalled read is retried, so keep attempts low.
    read_timeout=900,
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

SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    "cohere.embed-v4:0": "Cohere Embed v4",
    "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    "amazon.titan-embed-text-v2:0": "Titan Embeddings G2 - Text",
    # Disable Titan embedding.
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
    "amazon.nova-2-multimodal-embeddings-v1:0": "Nova Multimodal Embeddings V2",
}

ENCODER = tiktoken.get_encoding("cl100k_base")

# Global mapping: Profile ID/ARN → Foundation Model ID
# Handles both SYSTEM_DEFINED (cross-region) and APPLICATION profiles
# This enables feature detection for all profile types without pattern matching
profile_metadata = {}

# Models that don't support both temperature and topP simultaneously
# When both are provided, temperature takes precedence and topP is removed
TEMPERATURE_TOPP_CONFLICT_MODELS = {
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-5",
}

# Models that reject temperature and topP outright ("`temperature` is deprecated for this model")
# Both parameters are dropped from the request for these models
NO_SAMPLING_PARAMS_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-fable-5",
}

# Models that only accept adaptive thinking ("thinking.type.enabled" is rejected)
# reasoning_effort is mapped to output_config.effort instead of a budget_tokens value
ADAPTIVE_THINKING_MODELS = NO_SAMPLING_PARAMS_MODELS

# Models that reject forced tool choice ("tool_choice: type "tool" and "any" are not
# supported"). tool_choice "required" or a named function falls back to auto for them.
NO_FORCED_TOOL_CHOICE_MODELS = {
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5",
}

# Bedrock rejects Claude thinking budgets below this value
MIN_THINKING_BUDGET_TOKENS = 1024

# Remote image urls fetched for a single request. Each fetch has its own deadline,
# so this bounds how long one request can spend downloading. Converse accepts at
# most 20 images per request for most models anyway.
MAX_REMOTE_IMAGES_PER_REQUEST = 20

# Models that don't support assistant message prefill
# For these models, if conversation ends with assistant message (e.g., "continue response"),
# a user message will be added to ask the model to continue
NO_ASSISTANT_PREFILL_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-fable-5",
}


def list_bedrock_models() -> dict:
    """Automatically getting a list of supported models.

    Returns a model list combines:
        - ON_DEMAND models.
        - Cross-Region Inference Profiles (if enabled via Env)
        - Application Inference Profiles (if enabled via Env)
    """
    global profile_metadata
    model_list = {}
    metadata = {}
    try:
        if ENABLE_CROSS_REGION_INFERENCE:
            # List system defined inference profile IDs and store underlying model mapping
            paginator = bedrock_client.get_paginator('list_inference_profiles')
            for page in paginator.paginate(maxResults=1000, typeEquals="SYSTEM_DEFINED"):
                for profile in page["inferenceProfileSummaries"]:
                    profile_id = profile.get("inferenceProfileId")
                    if not profile_id:
                        continue

                    # Extract underlying model from first model in the profile
                    models = profile.get("models", [])
                    if models:
                        model_arn = models[0].get("modelArn", "")
                        if model_arn:
                            # Extract foundation model ID from ARN
                            model_id = model_arn.split('/')[-1]
                            metadata[profile_id] = {
                                "underlying_model_id": model_id,
                                "profile_type": "SYSTEM_DEFINED",
                            }

        if ENABLE_APPLICATION_INFERENCE_PROFILES:
            # List application defined inference profile IDs and create mapping
            paginator = bedrock_client.get_paginator('list_inference_profiles')
            for page in paginator.paginate(maxResults=1000, typeEquals="APPLICATION"):
                for profile in page["inferenceProfileSummaries"]:
                    try:
                        profile_arn = profile.get("inferenceProfileArn")
                        if not profile_arn:
                            continue

                        # Process all models in the profile
                        models = profile.get("models", [])
                        if not models:
                            logger.warning(f"Application profile {profile_arn} has no models")
                            continue

                        # Take first model - all models in array are same type (regional instances)
                        first_model = models[0]
                        model_arn = first_model.get("modelArn", "")
                        if not model_arn:
                            continue

                        # Extract model ID from ARN (works for both foundation models and cross-region profiles)
                        model_id = model_arn.split('/')[-1] if '/' in model_arn else model_arn

                        # Store in unified profile metadata for feature detection
                        metadata[profile_arn] = {
                            "underlying_model_id": model_id,
                            "profile_type": "APPLICATION",
                            "profile_name": profile.get("inferenceProfileName", ""),
                        }
                    except Exception as e:
                        logger.warning(f"Error processing application profile: {e}")
                        continue

        # List foundation models, only cares about text outputs here.
        response = bedrock_client.list_foundation_models(byOutputModality="TEXT")

        for model in response["modelSummaries"]:
            model_id = model.get("modelId", "N/A")
            stream_supported = model.get("responseStreamingSupported", True)
            status = model["modelLifecycle"].get("status", "ACTIVE")

            # currently, use this to filter out rerank models and legacy models
            if not stream_supported or status not in ["ACTIVE", "LEGACY"]:
                continue

            inference_types = model.get("inferenceTypesSupported", [])
            input_modalities = model["inputModalities"]
            # Add on-demand model list
            if "ON_DEMAND" in inference_types:
                model_list[model_id] = {"modalities": input_modalities}

            # Add all inference profiles (cross-region and application) for this model
            for profile_id, profile in metadata.items():
                if profile.get("underlying_model_id") == model_id:
                    model_list[profile_id] = {"modalities": input_modalities}

    except Exception as e:
        logger.error("Unable to list models: %s", e)
        # Keep serving the last good list rather than dropping every model but the default.
        if bedrock_model_list:
            return bedrock_model_list
    else:
        # Swap in the new mapping in one step so readers never see a partial one.
        profile_metadata = metadata

    if not model_list:
        # In case stack not updated.
        model_list[DEFAULT_MODEL] = {"modalities": ["TEXT", "IMAGE"]}

    return model_list


# Initialize the model list.
bedrock_model_list: dict = {}
bedrock_model_list = list_bedrock_models()


class BedrockModel(BaseChatModel):
    def __init__(self):
        # Per-stream state; a new BedrockModel is created for every request.
        self.think_emitted = False
        self.tool_call_indexes: dict[int, int] = {}

    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        global bedrock_model_list
        bedrock_model_list = list_bedrock_models()
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in bedrock_model_list.keys():
            # Provide helpful error for application profiles
            if "application-inference-profile" in chat_request.model:
                error = (
                    f"Application profile {chat_request.model} not found. "
                    f"Available profiles can be listed via GET /models API. "
                    f"Ensure ENABLE_APPLICATION_INFERENCE_PROFILES=true and "
                    f"the profile exists in your AWS account."
                )
            else:
                error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"
            logger.error("Unsupported model: %s", chat_request.model)

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    def _resolve_to_foundation_model(self, model_id: str) -> str:
        """
        Resolve any model identifier to foundation model ID for feature detection.

        Handles:
        - Cross-region profiles (us.*, eu.*, apac.*, global.*)
        - Application profiles (arn:aws:bedrock:...:application-inference-profile/...)
        - Foundation models (pass through unchanged)

        No pattern matching needed - just dictionary lookup.
        Unknown identifiers pass through unchanged (graceful fallback).

        Args:
            model_id: Can be foundation model ID, cross-region profile, or app profile ARN

        Returns:
            Foundation model ID if mapping exists, otherwise original model_id
        """
        if model_id in profile_metadata:
            return profile_metadata[model_id]["underlying_model_id"]
        return model_id

    def _supports_prompt_caching(self, model_id: str) -> bool:
        """
        Check if model supports prompt caching based on model ID pattern.

        Uses pattern matching instead of hardcoded whitelist for better maintainability.
        Automatically supports new models following the naming convention.

        Supported models:
        - Claude: anthropic.claude-* (excluding very old versions)
        - Nova: amazon.nova-*

        Returns:
            bool: True if model supports prompt caching
        """
        # Resolve profile to underlying model for feature detection
        resolved_model = self._resolve_to_foundation_model(model_id)
        model_lower = resolved_model.lower()

        # Claude models pattern matching
        if "anthropic.claude" in model_lower:
            # Exclude very old models that don't support caching
            excluded_patterns = ["claude-instant", "claude-v1", "claude-v2"]
            if any(pattern in model_lower for pattern in excluded_patterns):
                return False
            return True

        # Nova models pattern matching
        if "amazon.nova" in model_lower:
            return True

        # Future providers can be added here
        # Example: if "provider.model-name" in model_lower: return True

        return False

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""
        if DEBUG:
            logger.info("Raw request: " + chat_request.model_dump_json())

            # Log profile resolution for debugging
            if chat_request.model in profile_metadata:
                resolved = self._resolve_to_foundation_model(chat_request.model)
                profile_type = profile_metadata[chat_request.model].get("profile_type", "UNKNOWN")
                logger.info(
                    f"Profile resolution: {chat_request.model} ({profile_type}) → {resolved}"
                )

        # Convert the OpenAI request to a Bedrock request. This can fetch remote
        # images, so it runs in the thread pool to keep the event loop free.
        try:
            args = await run_in_threadpool(self._parse_request, chat_request)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Tool call arguments must be valid JSON: {e}")
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        try:
            if stream:
                return await run_in_threadpool(bedrock_runtime.converse_stream, **args)
            return await run_in_threadpool(bedrock_runtime.converse, **args)
        except Exception as e:
            raise bedrock_error_to_http(e, chat_request.model)

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        message_id = self.generate_message_id()
        response = await self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        usage = response["usage"]

        # Extract all token counts
        output_tokens = usage["outputTokens"]
        total_tokens = usage["totalTokens"]
        finish_reason = response["stopReason"]

        # Extract prompt caching metrics if available
        cache_read_tokens = usage.get("cacheReadInputTokens", 0)
        cache_creation_tokens = usage.get("cacheWriteInputTokens", 0)

        # Calculate actual prompt tokens
        # Bedrock's totalTokens includes all: inputTokens + cacheRead + cacheWrite + outputTokens
        # So: prompt_tokens = totalTokens - outputTokens
        actual_prompt_tokens = total_tokens - output_tokens

        chat_response = self._create_response(
            model=chat_request.model,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=actual_prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

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
        self.think_emitted = False
        self.tool_call_indexes = {}
        reasoning_tokens = 0
        try:
            # Reading the next event blocks on the socket, so do it in the thread pool.
            async for chunk in iterate_in_threadpool(stream):
                # Accumulate reasoning tokens from delta chunks before processing
                if "contentBlockDelta" in chunk:
                    delta = chunk["contentBlockDelta"].get("delta", {})
                    if "reasoningContent" in delta and "text" in delta["reasoningContent"]:
                        reasoning_tokens += len(ENCODER.encode(delta["reasoningContent"]["text"]))

                stream_response = self._create_response_stream(
                    model_id=chat_request.model, message_id=message_id, chunk=chunk
                )
                if not stream_response:
                    continue

                # Patch reasoning tokens into the final usage chunk
                if stream_response.usage and reasoning_tokens > 0:
                    stream_response.usage.completion_tokens_details = CompletionTokensDetails(
                        reasoning_tokens=reasoning_tokens,
                        audio_tokens=0,
                    )

                if DEBUG:
                    logger.info("Proxy response :" + stream_response.model_dump_json())
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

    @staticmethod
    def _prompt_caching_override(chat_request: ChatRequest, key: str) -> bool | None:
        """Return the per-request prompt caching override for ``key`` ("system" or "messages"), if any."""
        if not isinstance(chat_request.extra_body, dict):
            return None
        prompt_caching = chat_request.extra_body.get("prompt_caching")
        if not isinstance(prompt_caching, dict) or key not in prompt_caching:
            return None
        return prompt_caching[key] is True

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts with optional prompt caching support.

        Prompt caching can be enabled via:
        1. ENABLE_PROMPT_CACHING environment variable (global default)
        2. extra_body.prompt_caching.system = True/False (per-request override)

        Only adds cachePoint if:
        - Model supports caching (Claude, Nova)
        - Caching is enabled (ENV or extra_body)
        - System prompts exist

        Example output: [{"text" : system_prompt}, {"cachePoint": {"type": "default"}}]

        See: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        """
        system_prompts = []
        for message in chat_request.messages:
            if message.role not in ("system", "developer"):
                continue
            if isinstance(message.content, str):
                text = message.content
            else:
                text = "\n".join(part.text for part in message.content)
            system_prompts.append({"text": text})

        if not system_prompts:
            return system_prompts

        # Check if model supports prompt caching
        if not self._supports_prompt_caching(chat_request.model):
            return system_prompts

        # extra_body.prompt_caching.system overrides the ENABLE_PROMPT_CACHING default
        override = self._prompt_caching_override(chat_request, "system")
        cache_enabled = ENABLE_PROMPT_CACHING if override is None else override
        if not cache_enabled:
            return system_prompts

        # Add cache checkpoint after system prompts
        system_prompts.append({"cachePoint": {"type": "default"}})

        if DEBUG:
            logger.info(f"Added cachePoint to system prompts for model {chat_request.model}")

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                # Check if message has content that's not empty
                has_content = False
                if isinstance(message.content, str):
                    has_content = message.content.strip() != ""
                elif isinstance(message.content, list):
                    has_content = len(message.content) > 0
                elif message.content is not None:
                    has_content = True

                if has_content:
                    # Text message
                    messages.append(
                        {
                            "role": message.role,
                            "content": self._parse_content_parts(
                                message, chat_request.model
                            ),
                        }
                    )
                if message.tool_calls:
                    # Tool use message
                    for tool_call in message.tool_calls:
                        arguments = tool_call.function.arguments
                        tool_input = json.loads(arguments) if arguments and arguments.strip() else {}
                        messages.append(
                            {
                                "role": message.role,
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": tool_call.id,
                                            "name": tool_call.function.name,
                                            "input": tool_input,
                                        }
                                    }
                                ],
                            }
                        )
            elif isinstance(message, ToolMessage):
                # Bedrock does not support tool role,
                # Add toolResult to content
                # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html

                # Handle different content formats from OpenAI SDK
                tool_content = self._extract_tool_content(message.content)

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.tool_call_id,
                                    "content": [{"text": tool_content}],
                                }
                            }
                        ],
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return self._reframe_multi_payloard(messages, chat_request)

    def _extract_tool_content(self, content) -> str:
        """Extract text content from various OpenAI SDK tool message formats.

        Handles:
        - String content (legacy format)
        - List of content objects (OpenAI SDK 1.91.0+)
        - Nested JSON structures within text content
        """
        try:
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                text_parts = []
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        # Handle dict with 'text' field
                        if "text" in item:
                            item_text = item["text"]
                            if isinstance(item_text, str):
                                # Try to parse as JSON if it looks like JSON
                                if item_text.strip().startswith('{') and item_text.strip().endswith('}'):
                                    try:
                                        parsed_json = json.loads(item_text)
                                        # Convert JSON object to readable text
                                        text_parts.append(json.dumps(parsed_json, indent=2))
                                    except json.JSONDecodeError:
                                        # Silently fallback to original text
                                        text_parts.append(item_text)
                                else:
                                    text_parts.append(item_text)
                            else:
                                text_parts.append(str(item_text))
                        else:
                            # Handle other dict formats - convert to JSON string
                            text_parts.append(json.dumps(item, indent=2))
                    elif hasattr(item, 'text'):
                        # Handle ToolContent objects
                        text_parts.append(item.text)
                    else:
                        # Convert any other type to string
                        text_parts.append(str(item))
                return "\n".join(text_parts)

            # Fallback for any other type
            return str(content)
        except Exception as e:
            logger.warning("Tool content extraction failed: %s", str(e))
            # Return a safe fallback
            return str(content) if content is not None else ""

    def _reframe_multi_payloard(self, messages: list, chat_request: ChatRequest = None) -> list:
        """Receive messages and reformat them to comply with the Claude format

        With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
        with Claude format requests, you cannot repeatedly receive messages from the same role.

        This method searches through the OpenAI format messages in order and reformats them to the Claude format.

        ```
        openai_format_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Who are you?"},
        ]

        bedrock_format_messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Hello"},
                    {"text": "Who are you?"}
                ]
            },
        ]
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message["role"]
            next_content = message["content"]

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append(
                        {"role": current_role, "content": current_content}
                    )
                # Switch to the new role
                current_role = next_role
                current_content = []

            # Add the message content to current_content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)

        # Add the last role's messages to the list
        if current_content:
            reformatted_messages.append(
                {"role": current_role, "content": current_content}
            )

        # Bedrock Converse API requires conversations to end with a user message.
        # Some models don't support "assistant message prefill".
        # If the conversation ends with an assistant message (e.g., "continue response" scenario),
        # add a user message asking to continue - but only for models in NO_ASSISTANT_PREFILL_MODELS.
        if chat_request and reformatted_messages and reformatted_messages[-1]["role"] == "assistant":
            # Resolve profile to underlying model for feature detection
            resolved_model = self._resolve_to_foundation_model(chat_request.model)
            model_lower = resolved_model.lower()

            # Check if model is in the no-prefill list
            if any(no_prefill_model in model_lower for no_prefill_model in NO_ASSISTANT_PREFILL_MODELS):
                reformatted_messages.append({
                    "role": "user",
                    "content": [{"text": "Please continue your response from where you left off."}]
                })
                if DEBUG:
                    logger.info(f"Added continuation prompt for {chat_request.model} - conversation ended with assistant message")

        # Add cachePoint to messages if enabled and supported
        if chat_request and reformatted_messages:
            if not self._supports_prompt_caching(chat_request.model):
                return reformatted_messages

            # extra_body.prompt_caching.messages overrides the ENABLE_PROMPT_CACHING default
            override = self._prompt_caching_override(chat_request, "messages")
            cache_enabled = ENABLE_PROMPT_CACHING if override is None else override

            if cache_enabled:
                # Add cachePoint to the last user message content
                for msg in reversed(reformatted_messages):
                    if msg["role"] == "user" and msg.get("content"):
                        # Add cachePoint at the end of user message content
                        msg["content"].append({"cachePoint": {"type": "default"}})
                        if DEBUG:
                            logger.info(f"Added cachePoint to last user message for model {chat_request.model}")
                        break

        return reformatted_messages

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        remote_images = sum(
            1
            for message in chat_request.messages
            if isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, list)
            for part in message.content
            if isinstance(part, ImageContent) and not part.image_url.url.startswith("data:")
        )
        if remote_images > MAX_REMOTE_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"A request may reference at most {MAX_REMOTE_IMAGES_PER_REQUEST} remote image urls",
            )
        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        effective_max_tokens = (
            chat_request.max_completion_tokens
            if chat_request.max_completion_tokens is not None
            else chat_request.max_tokens
        )

        # Base inference parameters.
        inference_config = {}
        if effective_max_tokens is not None:
            inference_config["maxTokens"] = effective_max_tokens

        # Only include optional parameters when specified
        if chat_request.temperature is not None:
            inference_config["temperature"] = chat_request.temperature
        if chat_request.top_p is not None:
            inference_config["topP"] = chat_request.top_p

        # Some models (Claude Sonnet 4.5, Claude Sonnet 4.6, Haiku 4.5) don't support both temperature and topP
        # When both are provided, keep temperature and remove topP
        # Resolve profile to underlying model for feature detection
        resolved_model = self._resolve_to_foundation_model(chat_request.model)
        model_lower = resolved_model.lower()

        # Check if model is in the conflict list and both parameters are present
        if "temperature" in inference_config and "topP" in inference_config:
            if any(conflict_model in model_lower for conflict_model in TEMPERATURE_TOPP_CONFLICT_MODELS):
                inference_config.pop("topP", None)
                if DEBUG:
                    logger.info(f"Removed topP for {chat_request.model} (conflicts with temperature)")

        if any(model in model_lower for model in NO_SAMPLING_PARAMS_MODELS):
            inference_config.pop("temperature", None)
            inference_config.pop("topP", None)

        if chat_request.stop is not None:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            inference_config["stopSequences"] = stop

        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        if chat_request.reasoning_effort:
            # reasoning_effort is supported by Claude and DeepSeek v3
            # Different models use different formats
            if "anthropic.claude" in model_lower and any(
                model in model_lower for model in ADAPTIVE_THINKING_MODELS
            ):
                # Newer Claude models: adaptive thinking, depth controlled by effort
                args["additionalModelRequestFields"] = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": chat_request.reasoning_effort},
                }
            elif "anthropic.claude" in model_lower:
                # Claude format: reasoning_config = object with budget_tokens
                if effective_max_tokens is None:
                    raise HTTPException(
                        status_code=400,
                        detail="max_tokens or max_completion_tokens must be set when using reasoning_effort with Claude models",
                    )
                # unset topP - Not supported
                inference_config.pop("topP", None)

                budget_tokens = max(
                    MIN_THINKING_BUDGET_TOKENS,
                    self._calc_budget_tokens(effective_max_tokens, chat_request.reasoning_effort),
                )
                if budget_tokens >= effective_max_tokens:
                    raise HTTPException(
                        status_code=400,
                        detail=f"max_tokens must be greater than {MIN_THINKING_BUDGET_TOKENS} "
                        "when using reasoning_effort with this model",
                    )
                args["additionalModelRequestFields"] = {
                    "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
                }
            elif "deepseek.v3" in model_lower or "deepseek.deepseek-v3" in model_lower:
                # DeepSeek v3 format: reasoning_config = string ('low', 'medium', 'high')
                # From Bedrock Playground: {"reasoning_config": "high"}
                args["additionalModelRequestFields"] = {
                    "reasoning_config": chat_request.reasoning_effort  # Direct string: low/medium/high
                }
                if DEBUG:
                    logger.info(f"Applied reasoning_config={chat_request.reasoning_effort} for DeepSeek v3")
            else:
                # For other models (Qwen, etc.), ignore reasoning_effort parameter
                if DEBUG:
                    logger.info(f"reasoning_effort parameter ignored for model {chat_request.model} (not supported)")
        # add tool config
        # Converse has no "none" tool choice. Leaving tools out is the closest match, but
        # Bedrock requires the tool config whenever the history contains tool blocks.
        skip_tools = chat_request.tool_choice == "none" and not self._has_tool_blocks(messages)
        if chat_request.tools and not skip_tools:
            tool_config = {"tools": [self._convert_tool_spec(t.function) for t in chat_request.tools]}

            # Llama 3.1 rejects toolChoice; "none" cannot be expressed, so it falls back to auto.
            if (
                chat_request.tool_choice
                and chat_request.tool_choice != "none"
                and "meta.llama3-1-" not in model_lower
            ):
                if isinstance(chat_request.tool_choice, str):
                    # auto (default) is mapped to {"auto" : {}}
                    # required is mapped to {"any" : {}}
                    if chat_request.tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    if not isinstance(chat_request.tool_choice, dict) or "function" not in chat_request.tool_choice:
                        raise HTTPException(
                            status_code=400,
                            detail="tool_choice must contain 'function' key when specifying a specific tool",
                        )
                    tool_config["toolChoice"] = {"tool": {"name": chat_request.tool_choice["function"].get("name", "")}}

                forced = "any" in tool_config["toolChoice"] or "tool" in tool_config["toolChoice"]
                if forced and any(model in model_lower for model in NO_FORCED_TOOL_CHOICE_MODELS):
                    logger.warning("%s does not support forced tool choice, using auto", chat_request.model)
                    tool_config["toolChoice"] = {"auto": {}}
            args["toolConfig"] = tool_config
        # Add additional fields to enable extend thinking or other model-specific features
        if chat_request.extra_body:
            # Filter out prompt_caching (our control field, not for Bedrock)
            additional_fields = {
                k: v for k, v in chat_request.extra_body.items()
                if k != "prompt_caching"
            }

            if additional_fields:
                # Merge with existing additionalModelRequestFields (e.g., from reasoning_effort)
                existing = args.get("additionalModelRequestFields", {})
                existing.update(additional_fields)
                args["additionalModelRequestFields"] = existing

                # Extended thinking doesn't support both temperature and topP
                # Remove topP to avoid validation error
                if "thinking" in additional_fields:
                    inference_config.pop("topP", None)

        return args

    @staticmethod
    def _has_tool_blocks(messages: list[dict]) -> bool:
        return any(
            "toolUse" in block or "toolResult" in block
            for message in messages
            for block in message["content"]
        )

    def _estimate_reasoning_tokens(self, content: list[dict]) -> int:
        """
        Estimate reasoning tokens from reasoningContent blocks.

        Bedrock doesn't separately report reasoning tokens, so we estimate
        them using tiktoken to maintain OpenAI API compatibility.
        """
        reasoning_text = ""
        for block in content:
            if "reasoningContent" in block:
                # Some models (e.g. OpenAI GPT) return only encrypted redactedContent
                reasoning_text += block["reasoningContent"].get("reasoningText", {}).get("text", "")

        if reasoning_text:
            # Use tiktoken to estimate token count
            return len(ENCODER.encode(reasoning_text))
        return 0

    def _create_response(
        self,
        model: str,
        message_id: str,
        content: list[dict] | None = None,
        finish_reason: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> ChatResponse:
        message = ChatResponseMessage(
            role="assistant",
        )
        # A response can hold several text blocks, reasoning, and tool calls together.
        text_parts = []
        reasoning_parts = []
        tool_calls = []
        for part in content or []:
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
                        function=ResponseFunction(
                            name=tool["name"],
                            arguments=json.dumps(tool["input"]),
                        ),
                    )
                )
            else:
                logger.warning("Unknown tag in message content " + ",".join(part.keys()))

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        if reasoning:
            text = f"<think>{reasoning}</think>{text}"
        if tool_calls:
            message.tool_calls = tool_calls
            message.content = text or None
        else:
            message.content = text

        # Create prompt_tokens_details if cache metrics are available
        prompt_tokens_details = None
        if cache_read_tokens > 0 or cache_creation_tokens > 0:
            # Map Bedrock cache metrics to OpenAI format
            # cached_tokens represents tokens read from cache (cache hits)
            prompt_tokens_details = PromptTokensDetails(
                cached_tokens=cache_read_tokens,
                audio_tokens=0,
            )

        # Create completion_tokens_details if reasoning content exists
        completion_tokens_details = None
        reasoning_tokens = self._estimate_reasoning_tokens(content) if content else 0
        if reasoning_tokens > 0:
            completion_tokens_details = CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                audio_tokens=0,
            )

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
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens if total_tokens > 0 else input_tokens + output_tokens,
                prompt_tokens_details=prompt_tokens_details,
                completion_tokens_details=completion_tokens_details,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
        self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        if DEBUG:
            logger.info("Bedrock response chunk: " + str(chunk))

        finish_reason = None
        message = None
        usage = None

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

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                usage_data = metadata["usage"]

                # Extract prompt caching metrics if available
                cache_read_tokens = usage_data.get("cacheReadInputTokens", 0)
                cache_creation_tokens = usage_data.get("cacheWriteInputTokens", 0)

                # Create prompt_tokens_details if cache metrics are available
                prompt_tokens_details = None
                if cache_read_tokens > 0 or cache_creation_tokens > 0:
                    prompt_tokens_details = PromptTokensDetails(
                        cached_tokens=cache_read_tokens,
                        audio_tokens=0,
                    )

                # Calculate actual prompt tokens
                # Bedrock's totalTokens includes all tokens
                # prompt_tokens = totalTokens - outputTokens
                total_tokens = usage_data["totalTokens"]
                output_tokens = usage_data["outputTokens"]
                actual_prompt_tokens = total_tokens - output_tokens

                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=actual_prompt_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=total_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    ),
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
                usage=usage,
            )

        return None

    def _close_think_tag(self) -> str | None:
        """Return the closing reasoning tag if a reasoning block is open, and mark it closed."""
        if not self.think_emitted:
            return None
        self.think_emitted = False
        return "</think>"

    def _parse_image(self, image_url: str) -> tuple[bytes, str]:
        """Try to get the raw data from an image url.

        Accepts an inline base64 data url, or a remote http(s) url which the
        gateway validates before fetching so that a caller cannot reach internal
        endpoints through the proxy. Only supports 'image/jpeg', 'image/png',
        'image/gif' or 'image/webp'.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
        returns a tuple of (Image Data, Content Type)
        """
        return parse_image_url(image_url)

    def _parse_content_parts(
        self,
        message: UserMessage | AssistantMessage,
        model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self.is_supported_modality(model_id, modality="IMAGE"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        model = bedrock_model_list.get(model_id, {})
        modalities = model.get("modalities", [])
        if modality in modalities:
            return True
        return False

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

    def _calc_budget_tokens(
        self, max_tokens: int, reasoning_effort: Literal["low", "medium", "high"]
    ) -> int:
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
                "finished": "stop",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "complete": "stop",
                "content_filtered": "content_filter",
            }
            return finish_reason_mapping.get(
                finish_reason.lower(), finish_reason.lower()
            )
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"
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

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        try:
            return bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except Exception as e:
            raise bedrock_error_to_http(e, model_id)

    def _create_response(
        self,
        embeddings: list[list[float]],
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
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
            response = self._invoke_model(
                args=self._parse_args(texts[start : start + self.BATCH_SIZE], dimensions),
                model_id=embeddings_request.model,
            )
            response_body = json.loads(response.get("body").read())
            if DEBUG:
                logger.info("Bedrock response body: " + str(response_body))
            embeddings = response_body["embeddings"]
            # v4 (and v3 with embedding_types) returns {"float": [...]} instead of a list.
            all_embeddings.extend(embeddings["float"] if isinstance(embeddings, dict) else embeddings)
            # Cohere reports usage only in the response headers.
            headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
            total_tokens += int(headers.get("x-amzn-bedrock-input-token-count", 0))

        return self._create_response(
            embeddings=all_embeddings,
            model=embeddings_request.model,
            input_tokens=total_tokens,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):
    # Titan Text Embeddings V2 output sizes; V1 has a fixed size.
    V2_MODEL_ID = "amazon.titan-embed-text-v2:0"
    V2_DIMENSIONS = {256, 512, 1024}

    def _parse_args(self, input_text: str, dimensions: int | None) -> dict:
        args = {"inputText": input_text}
        if dimensions is not None:
            args["dimensions"] = dimensions
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        dimensions = embeddings_request.dimensions
        if not embeddings_request.model.endswith(self.V2_MODEL_ID):
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
            response = self._invoke_model(
                args=self._parse_args(text, dimensions),
                model_id=embeddings_request.model,
            )
            response_body = json.loads(response.get("body").read())
            if DEBUG:
                logger.info("Bedrock response body: " + str(response_body))
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
            response = self._invoke_model(
                args=self._parse_args(text, dimension),
                model_id=embeddings_request.model,
            )
            response_body = json.loads(response.get("body").read())
            if DEBUG:
                logger.info("Bedrock response body keys: " + str(list(response_body.keys())))

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


# Inference profile prefixes, e.g. global.cohere.embed-v4:0 for a model only served through a profile.
_PROFILE_PREFIXES = ("global.", "us.", "eu.", "apac.", "jp.", "au.", "ca.", "us-gov.")


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    base_model_id = model_id
    for prefix in _PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            base_model_id = model_id[len(prefix) :]
            break
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(base_model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    match model_name:
        case "Cohere Embed Multilingual" | "Cohere Embed English":
            return CohereEmbeddingsModel()
        case "Cohere Embed v4":
            return CohereEmbeddingsModel(is_v4=True)
        case "Titan Embeddings G1 - Text" | "Titan Embeddings G2 - Text":
            return TitanEmbeddingsModel()
        case "Nova Multimodal Embeddings V2":
            return NovaEmbeddingsModel()
        case _:
            logger.error("Unsupported model id " + model_id)
            raise HTTPException(
                status_code=400,
                detail="Unsupported embedding model id " + model_id,
            )
