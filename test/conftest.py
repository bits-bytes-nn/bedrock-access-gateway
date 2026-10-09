import os
import sys
from unittest import mock

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# api.auth refuses to import without an API key, and api.setting reads its
# defaults at import time, so both are set before any api module is imported.
os.environ.setdefault("API_KEY", "test-api-key")

# api.models.bedrock creates boto3 clients and lists models at import time. Replace
# the clients with mocks so the tests never reach AWS, whatever credentials are set.
boto3.client = lambda *args, **kwargs: mock.MagicMock(name=f"boto3.client({args[0] if args else kwargs.get('service_name')})")
