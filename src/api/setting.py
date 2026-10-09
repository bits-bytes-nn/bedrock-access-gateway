import os


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable, refusing values that are neither true nor false."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false, got {value!r}")


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        number = int(value)
    except ValueError:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    if number <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    return number


API_ROUTE_PREFIX = os.environ.get("API_ROUTE_PREFIX", "/api/v1")

TITLE = "Amazon Bedrock Proxy APIs"
SUMMARY = "OpenAI-Compatible RESTful APIs for Amazon Bedrock"
VERSION = "0.1.0"
DESCRIPTION = """
Use OpenAI-Compatible RESTful APIs for Amazon Bedrock models.
"""

DEBUG = _env_flag("DEBUG", False)
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
ENABLE_CROSS_REGION_INFERENCE = _env_flag("ENABLE_CROSS_REGION_INFERENCE", True)
ENABLE_APPLICATION_INFERENCE_PROFILES = _env_flag("ENABLE_APPLICATION_INFERENCE_PROFILES", True)
ENABLE_PROMPT_CACHING = _env_flag("ENABLE_PROMPT_CACHING", False)

# Multimodal image inputs may reference a remote URL that the gateway fetches on
# behalf of the caller. See api/image_url.py for the checks applied to that URL.
ENABLE_IMAGE_URL_FETCH = _env_flag("ENABLE_IMAGE_URL_FETCH", True)
# Optional comma-separated allowlist of hosts an image url may point at.
# Empty means any host that resolves to a globally routable address is allowed.
IMAGE_URL_ALLOWED_HOSTS = frozenset(
    host.strip().lower() for host in os.environ.get("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if host.strip()
)
IMAGE_URL_MAX_SIZE_MB = _env_positive_int("IMAGE_URL_MAX_SIZE_MB", 10)
