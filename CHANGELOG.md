# Changelog

Changes in this fork since it diverged from upstream at `144c4bc` (upstream deprecation notice).

## Unreleased

### Added
- `response_format` `json_schema` is sent to Bedrock as structured outputs (`outputConfig.textFormat`);
  `json_object` adds a JSON instruction to the system prompt. JSON responses carry no `<think>` block.
- Fargate template: optional HTTPS listener (`CertificateArn`) with HTTP redirected to HTTPS, container logs in
  `/ecs/<stack name>-proxy-api`, a deployment circuit breaker, and the `DefaultEmbeddingModelId`,
  `LogRetentionInDays`, `EnableImageUrlFetch` and `ImageUrlAllowedHosts` parameters.
- API Gateway stage throttling, set by the `ThrottlingRateLimit` (default 10 requests per second) and
  `ThrottlingBurstLimit` (default 20) stack parameters, to bound the spend if the API key leaks.
- `IMAGE_URI` for `scripts/deploy-lambda.sh` redeploys a pushed image by digest, for rollbacks.
- `src/requirements.in`; `src/requirements.txt` is now a full lock including transitive dependencies.
- CI lints the template (cfn-lint) and scripts (shellcheck) and builds the Lambda image.
- `src/api/models/model_rules.py`: one table of per-model request rules, keyed by Claude family and version.
  Future Claude versions inherit the newest rules.
- Model list refresh when a request names a model that is not listed (at most once a minute), so a model
  enabled after cold start, or a failed first listing, no longer requires a call to `GET /models`.
- Images in tool results are sent to the model as images.
- `scripts/deploy-lambda.sh`: non-interactive build, push and stack update that pins the image digest.
- Stack parameters `EnableImageUrlFetch` and `ImageUrlAllowedHosts`.
- `requirements-dev.txt`, pytest configuration and a CI workflow (ruff and pytest).

### Changed
- With extended thinking (Claude 3.7 Sonnet to Opus 4.6), `temperature` is dropped and a forced tool choice
  falls back to `auto`; Bedrock rejected both combinations. Claude 3.5 and earlier ignore `reasoning_effort`
  instead of failing.
- A leading `<think>` block in assistant history is removed before the conversation is sent to Bedrock.
- `GET /models/{id}` uses the cached model list, refreshing it (at most once a minute) only for unknown ids.
- CORS no longer allows credentials; the API key is sent in the `Authorization` header.
- `ApiKeySecretArn` must be the full secret ARN, which the IAM policy needs to match.
- Fargate template: default model `global.anthropic.claude-opus-5-5` and embedding model
  `amazon.titan-embed-text-v2:0` (previously the retired Claude 3 Sonnet and a fixed Cohere v3), no
  `secretsmanager:DescribeSecret`, log permissions scoped to the log group, and partition-aware ARNs.
- Prompt caching is only applied to models that support it (Claude 3.7 Sonnet, Claude 3.5 Haiku and later,
  Nova). Claude 3 Haiku and 3.5 Sonnet previously failed every request while caching was enabled.
- Claude Opus 4.6 no longer receives `temperature` and `top_p` together.
- `stream_options.include_usage` defaults to `false`, as in the OpenAI API.
- `n` greater than 1 returns `400`; `response_format` other than `text` is logged and ignored.
- Bedrock stop reasons outside the OpenAI set are mapped (`guardrail_intervened` to `content_filter`,
  `model_context_window_exceeded` to `length`, others to `stop`).
- Default model `global.anthropic.claude-opus-5-5`; the Bedrock read timeout is 550 seconds, below the
  Lambda and API Gateway limits.
- The Lambda role no longer has `secretsmanager:DescribeSecret`, and Bedrock ARNs use the stack partition.

### Fixed
- Empty text blocks (system prompt, assistant text) are no longer sent; Converse rejects them.
- Tool result text is passed through unchanged instead of being re-indented.

### Removed
- `mangum`, unused with the Lambda Web Adapter.
- The unused `reasoning_content` response field.

## 2026-10-10

### Security
- Remote image URLs are parsed once and fetched from the exact address that passed the checks, closing an
  SSRF bypass through URLs that parsers read differently (`http://169.254.169.254\@example.com/`) and DNS
  rebinding. Fetches have an overall deadline, and a request may reference at most 20 remote images.

### Added
- Request rules for Claude Opus 4.7/4.8 and the Claude 5 family (no sampling parameters, adaptive thinking,
  no assistant prefill) and for Claude Opus and Sonnet 5.5 (no forced tool choice).
- Cohere Embed v4, inference profile embedding ids, Titan Text V2 `dimensions`.
- Titan embeddings accept a list of inputs.
- Stack parameters `DefaultEmbeddingModelId` and `LogRetentionInDays`.

### Fixed
- Blocking calls (image fetches, stream reads, embeddings, model listing) run off the event loop.
- Streaming errors before the first event return an HTTP status; Bedrock errors map to matching statuses
  without exposing account details.
- OpenAI compatibility: tools without a description, tool call indexes, `finish_reason` with open reasoning,
  multiple text blocks, `tool_choice: none`, text-part system messages, token-array embedding inputs,
  Cohere batches over 96 inputs, encrypted reasoning blocks from OpenAI models.
- Settings are parsed strictly, API keys are compared in constant time.
