# Changelog

Changes in this fork since it diverged from upstream at `144c4bc` (upstream deprecation notice).

## Unreleased

### Added
- `src/api/models/model_rules.py`: one table of per-model request rules, keyed by Claude family and version.
  Future Claude versions inherit the newest rules.
- Model list refresh when a request names a model that is not listed (at most once a minute), so a model
  enabled after cold start, or a failed first listing, no longer requires a call to `GET /models`.
- Images in tool results are sent to the model as images.
- `scripts/deploy-lambda.sh`: non-interactive build, push and stack update that pins the image digest.
- Stack parameters `EnableImageUrlFetch` and `ImageUrlAllowedHosts`.
- `requirements-dev.txt`, pytest configuration and a CI workflow (ruff and pytest).

### Changed
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
