import hmac
import json
import os
from typing import Annotated

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

api_key_param = os.environ.get("API_KEY_PARAM_NAME")
api_key_secret_arn = os.environ.get("API_KEY_SECRET_ARN")
api_key_env = os.environ.get("API_KEY")
if api_key_param:
    # For backward compatibility.
    # Please now use secrets manager instead.
    ssm = boto3.client("ssm")
    api_key = ssm.get_parameter(Name=api_key_param, WithDecryption=True)["Parameter"]["Value"]
elif api_key_secret_arn:
    sm = boto3.client("secretsmanager")
    try:
        response = sm.get_secret_value(SecretId=api_key_secret_arn)
        api_key = json.loads(response["SecretString"])["api_key"]
    except ClientError as e:
        raise RuntimeError("Unable to retrieve API KEY, please ensure the secret ARN is correct") from e
    except (KeyError, TypeError, json.JSONDecodeError):
        # The secret value itself is deliberately not chained into the traceback.
        raise RuntimeError('Please ensure the secret is a JSON string with an "api_key" field') from None
elif api_key_env:
    api_key = api_key_env
else:
    raise RuntimeError(
        "API Key is not configured. Please set up your API Key."
    )

security = HTTPBearer()


async def api_key_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    # Constant-time comparison so response timing does not reveal the key.
    if not hmac.compare_digest(credentials.credentials.encode(), api_key.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
