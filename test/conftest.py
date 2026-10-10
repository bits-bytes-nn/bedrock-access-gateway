import os
from unittest import mock

import boto3

# The api modules read their settings at import time, so the environment is fixed
# here, before any of them is imported. Variables exported in the developer's shell
# (an API key, an image url allowlist, ...) would otherwise change test results.
_GATEWAY_SETTINGS = (
    "ALLOWED_ORIGINS",
    "API_KEY",
    "API_KEY_PARAM_NAME",
    "API_KEY_SECRET_ARN",
    "API_ROUTE_PREFIX",
    "DEBUG",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL",
    "ENABLE_APPLICATION_INFERENCE_PROFILES",
    "ENABLE_CROSS_REGION_INFERENCE",
    "ENABLE_IMAGE_URL_FETCH",
    "ENABLE_PROMPT_CACHING",
    "IMAGE_URL_ALLOWED_HOSTS",
    "IMAGE_URL_MAX_SIZE_MB",
)
for _name in _GATEWAY_SETTINGS:
    os.environ.pop(_name, None)
os.environ["API_KEY"] = "test-api-key"
os.environ["AWS_REGION"] = "us-west-2"

# api.models.bedrock creates boto3 clients and lists models at import time. Replace
# the clients with mocks so the tests never reach AWS, whatever credentials are set.
boto3.client = lambda *args, **kwargs: mock.MagicMock(
    name=f"boto3.client({args[0] if args else kwargs.get('service_name')})"
)
