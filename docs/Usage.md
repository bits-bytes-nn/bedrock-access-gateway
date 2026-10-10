[中文](./Usage_CN.md) (may lag behind this English guide)

# Usage Guide

Assuming you have set up below environment variables after deployed:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

**API Example:**
- [Models API](#models-api)
- [Embedding API](#embedding-api)
- [Multimodal API](#multimodal-api)
- [Tool Call](#tool-call)
- [Structured output](#structured-output)
- [Reasoning](#reasoning)
- [Interleaved thinking (beta)](#interleaved-thinking-beta)

## Models API

You can use this API to get a list of supported model IDs.

Also, you can use this API to refresh the model list if new models are added to Amazon Bedrock.


**Example Request**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Example Response**

```bash
[
  ...
  {
    "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  {
    "id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  ...
]
```

## Chat Completions API

### Basic Example with Claude Sonnet 4.5

Claude Sonnet 4.5 is Anthropic's most intelligent model, excelling at coding, complex reasoning, and agent-based tasks. It's available via global cross-region inference profiles.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
      }
    ]
  }'
```

**Example SDK Usage**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming.",
        }
    ],
)

print(completion.choices[0].message.content)
```

## Embedding API

**Important Notice**: Please carefully review the following points before using this proxy API for embedding.

1. If you have previously used OpenAI embedding models to create vectors, be aware that switching to a new model may not be straightforward. Different models have varying dimensions (e.g., embed-multilingual-v3.0 has 1024 dimensions), and even for the same text, they may produce different results.
2. If you are using OpenAI embedding models for encoded integers (such as with LangChain), this solution will attempt to decode the integers using `tiktoken` to retrieve the original text. However, there is no guarantee that the decoded text will be accurate.
3. If you are using OpenAI embedding models for long texts, you should verify the maximum number of tokens supported for Bedrock models, e.g. for optimal performance, Bedrock recommends limiting the text length to less than 512 tokens.
4. `dimensions` is supported for Titan Text Embeddings V2 (256, 512, 1024), Cohere Embed v4 (256, 512, 1024, 1536) and Nova embeddings; other models return `400` when it is set. Inference profile IDs such as `global.cohere.embed-v4:0` are accepted, which is required in Regions where a model is only served through a profile.
5. Supported models: Amazon Titan Text Embeddings V1 and V2, Cohere Embed English/Multilingual v3 and v4, and Amazon Nova multimodal embeddings (text input only, default 3072 dimensions; 256, 384, 1024 or 3072). Availability differs by Region. OpenAI model names such as `text-embedding-3-small` use the `DEFAULT_EMBEDDING_MODEL` (Titan V2 by default). Cohere models embed inputs as `search_document`.


**Example Request**

```bash
curl $OPENAI_BASE_URL/embeddings \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "input": "The food was delicious and the waiter...",
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }'
```

**Example Response**

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.02279663,
                -0.024612427,
                0.012863159,
                ...
                0.01612854,
                0.0038928986
            ],
            "index": 0
        }
    ],
    "model": "cohere.embed-multilingual-v3",
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
}
```

Alternatively, you can use the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


text = "hello"
# will output like [0.003578186, 0.028717041, 0.031021118, -0.0014066696,...]
print(get_embedding(text))
```

Or LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:5])
doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])
```

## Multimodal API

> [!NOTE]
> When you pass a remote `image_url`, the gateway fetches it server-side. Only `http(s)` URLs that resolve
> to a publicly routable address are fetched; URLs pointing at the instance metadata service, `localhost`,
> or private ranges are rejected with a `400`. You can restrict this further with `IMAGE_URL_ALLOWED_HOSTS`
> or turn it off with `ENABLE_IMAGE_URL_FETCH=false` — see the [Security Guide](./Security.md#2-image-url-fetching-ssrf).

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in these images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/bits-bytes-nn/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

If you need to use this API with non-public images, you can do base64 the image first and pass the encoded string. 
Replace `image/jpeg` with the actual content type. Currently, only 'image/jpeg', 'image/png', 'image/gif' or 'image/webp' is supported.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in this images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,<your image data>"
                    }
                }
            ]
        }
    ]
}'
```

**Example Response**

```json
{
    "id": "msg_01BY3wcz41x7XrKhxY3VzWke",
    "created": 1712543069,
    "model": "global.anthropic.claude-opus-5-5",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The image contains the following objects:\n\n1. A peach-colored short-sleeve button-up shirt\n2. An olive green plaid long coat/jacket\n3. A pair of white sneakers or canvas shoes\n4. A brown shoulder bag or purse\n5. A makeup brush or cosmetic applicator\n6. A tube or container (possibly lipstick or lip balm)\n7. A pair of sunglasses\n8. A thought bubble icon\n9. A footprint icon\n10. A leaf or plant icon\n11. A flower icon\n12. A cloud icon\n\nIn total, there are 12 distinct objects depicted in the illustrated scene."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 197,
        "completion_tokens": 147,
        "total_tokens": 344
    }
}
```


## Tool Call

**Important Notice**: Please carefully review the following points before using this Tool Call for Chat completion API.

1. Function Call is now deprecated in favor of Tool Call by OpenAI, hence it's not supported here, you should use Tool Call instead.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Shanghai today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or state which is required."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "Use this tool to get the current location if user does not provide a location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ],
    "tool_choice": "auto"
}'
```

**Example Response**

```json
{
    "id": "chatcmpl-1a2b3c4d",
    "created": 1712543689,
    "model": "global.anthropic.claude-sonnet-5-5",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"Shanghai\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "total_tokens": 320
    }
}
```

You can try it with different questions, such as:
1. Hello, who are you?  (No tools are needed)
2. What is the weather like today?  (Should use get_current_location tool first)


## Structured output

`response_format` with `type: "json_schema"` is translated to the Converse
[structured outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html) setting, so
Bedrock constrains the answer to the schema. It works with the models that support structured outputs (checked: Claude
Haiku 4.5, Sonnet 5.5 and Opus 5.5, and OpenAI GPT-6); other models (for example Amazon Nova and Claude 3.5) return
`400 This model doesn't support the outputConfig field`.

Bedrock supports a subset of JSON Schema: objects need `"additionalProperties": false`, and numeric and string
length constraints are not supported. A new schema is compiled on first use, which can take a while; it is cached
for 24 hours. `strict` is accepted and has no effect, because the schema is always enforced.

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class City(BaseModel):
    name: str
    population: int


completion = client.chat.completions.parse(
    model="global.anthropic.claude-opus-5-5",
    messages=[{"role": "user", "content": "What is the largest city in Korea?"}],
    response_format=City,
)
print(completion.choices[0].message.parsed)
```

`type: "json_object"` cannot be enforced by Converse, which needs a schema. The gateway adds an instruction to
answer with a single JSON object to the system prompt, so the output is usually, but not always, valid JSON. Use
`json_schema` when the output must parse.

With either JSON format, reasoning is not added to the message content as a `<think>` block, so the content stays
parseable; reasoning tokens are still reported in `usage`. Strict tool definitions (`strict` on a function) are not
passed to Bedrock, because several models reject the field.

## Reasoning

Pass `reasoning_effort` (`low`, `medium` or `high`) to turn on reasoning for models that support it. How the
gateway expresses it depends on the model family:

| Models | What `reasoning_effort` becomes | Notes |
|---|---|---|
| Claude Opus 4.7, Opus 4.8 and Claude 5.x | Adaptive thinking: `thinking.type: adaptive` with `output_config.effort` | `max_tokens` is not required. These models reject `temperature` and `top_p`, so both are dropped. |
| Claude 3.7 Sonnet to Claude Opus 4.6 | Extended thinking with `budget_tokens` of 30% (low), 60% (medium) or 80% (high) of `max_tokens`, at least 1,024 | `max_tokens` (or `max_completion_tokens`) is required and must be above 1,024. While thinking, `temperature` and `top_p` are dropped and a forced tool choice falls back to `auto`, which Bedrock requires. |
| DeepSeek V3 | `reasoning_config` set to the effort string | |
| DeepSeek R1, OpenAI GPT, Kimi, GLM | Ignored; these models reason on their own | |
| Claude 3.5 and earlier | Ignored; no extended thinking | |

Other per-model adjustments the gateway makes (see [`model_rules.py`](../src/api/models/model_rules.py)):

- Claude 4.5 and 4.6 accept `temperature` or `top_p` but not both; `top_p` is dropped when both are set.
- Claude 4.6 and later do not accept a conversation that ends with an assistant message (prefill). The gateway
  appends a user message asking the model to continue.
- Claude Opus 5.5 and Sonnet 5.5 reject forced tool choice, so `tool_choice: "required"` or a named function
  falls back to `auto` for them.

The reasoning text is returned inside the message content, wrapped in `<think>` and `</think>`, followed by
the answer. In streaming responses the tags arrive as part of the content deltas. Models that return
encrypted reasoning (OpenAI GPT) contribute no `<think>` block. When a client sends an earlier answer back in the
conversation history, a leading `<think>...</think>` block is removed from it, so the model does not read its old
reasoning as answer text.

> **Note**: Omitting `max_tokens` (or `max_completion_tokens`) lets Bedrock use the model's native maximum
> output rather than a fixed cap. Set `max_tokens` explicitly if you need to control costs or latency.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "stream": false
}'
```

**Example Response**

```json
{
    "id": "chatcmpl-83fb7a88",
    "created": 1740545278,
    "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "<think>I need to compare the decimal numbers 3.9 and 3.11...</think>3.9 is bigger than 3.11..."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 51,
        "completion_tokens": 565,
        "total_tokens": 616,
        "completion_tokens_details": {"reasoning_tokens": 420, "audio_tokens": 0}
    }
}
```

With the OpenAI SDK (`pip3 install -U openai`), split the reasoning from the answer:

```python
import re

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="global.anthropic.claude-opus-5-5",
    messages=[{"role": "user", "content": "which one is bigger, 3.9 or 3.11?"}],
    reasoning_effort="low",
)

content = response.choices[0].message.content
match = re.match(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
reasoning, answer = match.groups() if match else ("", content)
```

When streaming, concatenate the content deltas and split the result the same way.

## Interleaved thinking (beta)

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.

Extended thinking with tool use in Claude 4 models supports [interleaved thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-tool-use-interleaved) enables Claude 4 models to think between tool calls and run more sophisticated reasoning after receiving tool results. which is helpful for more complex agentic interactions.
With interleaved thinking, the `budget_tokens` can exceed the `max_tokens` parameter because it represents the total budget across all thinking blocks within one assistant turn.

> **Note**: `thinking.type: enabled` with `budget_tokens` is rejected by Claude Opus 4.7, Opus 4.8 and Claude
> 5.x, which only accept adaptive thinking. For those models use `reasoning_effort` instead of the examples below.

**Supported Models**: Claude Sonnet 4, Claude Sonnet 4.5

**Example Request**

- Non-Streaming (Claude Sonnet 4.5)

```bash
curl $OPENAI_BASE_URL/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
"model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Non-Streaming (Claude Sonnet 4)

```bash
curl $OPENAI_BASE_URL/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
"model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Sonnet 4.5)

```bash
curl $OPENAI_BASE_URL/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
"model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Sonnet 4)

```bash
curl $OPENAI_BASE_URL/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
"model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```
