# Troubleshooting Guide

This guide helps you troubleshoot common issues you might encounter when using the Bedrock Access Gateway.

## Where to find logs

The Lambda function logs to the CloudWatch log group `/aws/lambda/<stack name>-proxy-api`, which the stack
creates with the `LogRetentionInDays` retention. Stacks deployed before that log group existed also have an
older `/aws/lambda/<stack name>-ProxyApiHandler-<id>` group.

```bash
aws logs tail /aws/lambda/<stack name>-proxy-api --since 1h --follow
```

## Error format

Errors are returned as JSON with a `detail` field and an HTTP status:

```json
{"detail": "Unsupported model nope, please use models API to get a list of supported models"}
```

Once a streaming response has started, an error can only be sent as an event inside the stream:

```
data: {"error": {"message": "..."}}
```

Request errors (unknown model, invalid parameters) are detected before the stream starts and return a normal
HTTP status.

| Status | Cause |
|---|---|
| `400` | Invalid request, unsupported model or parameter, or a Bedrock `ValidationException` (its message is included) |
| `401` / `403` | Missing or wrong API key (`Invalid API Key`, `Not authenticated`) |
| `403` | Bedrock denied access to the model. The message is generic; the full error, including the role, is in the logs |
| `404` | Unknown model id in `GET /models/{id}`, or a Bedrock `ResourceNotFoundException` |
| `413` | A remote image is larger than `IMAGE_URL_MAX_SIZE_MB` |
| `429` | Bedrock throttling or quota exceeded |
| `502` / `503` / `504` | Bedrock model error, model or service unavailable, or model timeout |
| `500` | Unexpected error; see the logs |

## Common issues

### The function fails to start

The application reads the API key when it starts and stops if it cannot:

- `API Key is not configured`: none of `API_KEY_SECRET_ARN`, `API_KEY_PARAM_NAME` or `API_KEY` is set.
- `Unable to retrieve API KEY`: the secret ARN is wrong, or the function role cannot read it.
- `Please ensure the secret is a JSON string with an "api_key" field`: store the secret as key/value with the
  key `api_key`.
- `... must be true or false` or `... must be a positive integer`: a setting has an invalid value, see
  [Configuration](../README.md#configuration).

### Unsupported model

The model must be listed by `GET /models`, which shows the foundation models and inference profiles available
in the Region. A model enabled after the function started is picked up automatically on the first request that
names it. Many current models are only available through an inference profile; use the profile id
(`global.anthropic.claude-opus-5-5`, `global.cohere.embed-v4:0`) rather than the foundation model id.

### Access denied (403)

- Make sure model access is enabled for the model in the Amazon Bedrock console.
- Some models need an extra opt-in. For example, Claude Fable requires a data retention setting and returns
  `data retention mode 'default' is not available for this model` until it is configured.

### Parameter errors from newer models

Newer models reject some OpenAI parameters. The gateway adjusts requests for the models it knows (see
[`src/api/models/model_rules.py`](../src/api/models/model_rules.py)): for example it drops `temperature` and
`top_p` for Claude 5 models. If a model returns an error such as `` `temperature` is deprecated for this model``,
its family may need a rule; see [Supporting a new model family](../README.md#supporting-a-new-model-family).
OpenAI GPT models on Bedrock reject `temperature` and `max_tokens`; use `max_completion_tokens`.

### Throttling (429)

- Check the Bedrock service quotas for the model in the AWS console and request an increase if needed.
- The gateway retries throttled calls a few times; add retries with backoff in your client for sustained load.

### Remote images are rejected

Remote image URLs are only fetched when they resolve to a publicly routable address, and URLs with
credentials, backslashes or whitespace are refused. See the [Security Guide](./Security.md). Send the image as a
base64 `data:` URL if it is not publicly reachable.

## Getting Help

If you're still experiencing issues:

1. Check the CloudWatch Logs for detailed error messages
2. Verify your AWS credentials and permissions
3. Review the [Usage Guide](./Usage.md) for correct API usage
4. Open an issue in this repository with the error message, steps to reproduce, the Region and model, and the
   relevant log lines

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Supported models and Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Amazon Bedrock quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html)
