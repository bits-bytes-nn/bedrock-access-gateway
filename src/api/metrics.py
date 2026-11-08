"""Per-request token metrics in CloudWatch embedded metric format (EMF).

Each completion or embeddings call writes one JSON line to stdout. Lambda sends stdout to
CloudWatch Logs, which extracts the metrics from lines that follow the EMF specification:
https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html

The line holds the model id and token counts only: no request or response content, API
key, client address or other caller detail.
"""

import json
import logging
import sys
import threading
import time

from api.setting import ENABLE_TOKEN_METRICS, TOKEN_METRICS_NAMESPACE

logger = logging.getLogger(__name__)

# Metric name -> Converse usage field. InputTokens excludes cache reads and writes, as in Bedrock.
_METRICS = {
    "InputTokens": "inputTokens",
    "OutputTokens": "outputTokens",
    "CacheReadTokens": "cacheReadInputTokens",
    "CacheWriteTokens": "cacheWriteInputTokens",
}
# Per model, and without dimensions: the dimensionless metric totals every model for alarms.
_DIMENSION_SETS = [["Model"], []]
# A dimension value must be 1 to 1024 printable ASCII characters, or EMF drops that dimension set.
_MAX_DIMENSION_LENGTH = 1024

# Lines are written whole under a lock, so concurrent requests cannot interleave them.
_lock = threading.Lock()


def _dimension_value(model_id: str) -> str:
    if 0 < len(model_id) <= _MAX_DIMENSION_LENGTH and all(" " <= char <= "~" for char in model_id):
        return model_id
    return "other"


def token_metrics_line(model_id: str, usage: dict) -> str:
    """Return the EMF line for one call's Converse-style usage (missing counts are 0)."""
    line = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": TOKEN_METRICS_NAMESPACE,
                    "Dimensions": _DIMENSION_SETS,
                    "Metrics": [{"Name": name, "Unit": "Count"} for name in _METRICS],
                }
            ],
        },
        "Model": _dimension_value(model_id),
    }
    for name, field in _METRICS.items():
        line[name] = int(usage.get(field) or 0)
    return json.dumps(line, separators=(",", ":"))


def record_tokens(model_id: str, usage: dict | None) -> None:
    """Write the token metrics of one call to stdout when ENABLE_TOKEN_METRICS is set.

    Never raises: a metrics failure must not fail the request it describes.
    """
    if not ENABLE_TOKEN_METRICS or usage is None:
        return
    try:
        line = token_metrics_line(model_id, usage)
        with _lock:
            sys.stdout.write(line + "\n")
            # stdout is block-buffered when it is not a terminal; flush so the line reaches
            # CloudWatch Logs with the request rather than when the buffer fills.
            sys.stdout.flush()
    except Exception:
        logger.warning("Could not write token metrics for %s", model_id, exc_info=True)
