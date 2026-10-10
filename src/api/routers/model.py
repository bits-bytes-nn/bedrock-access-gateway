from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from starlette.concurrency import run_in_threadpool

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel, is_model_listed
from api.schema import Model, Models

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
)

chat_model = BedrockModel()


async def validate_model_id(model_id: str):
    # Refreshing the list calls Bedrock with blocking boto3 calls.
    if not await run_in_threadpool(is_model_listed, model_id):
        raise HTTPException(status_code=404, detail="Unsupported Model Id")


@router.get("", response_model=Models)
async def list_models():
    model_list = [Model(id=model_id) for model_id in await run_in_threadpool(chat_model.list_models)]
    return Models(data=model_list)


@router.get(
    # :path so that application inference profile ARNs, which contain "/", match.
    "/{model_id:path}",
    response_model=Model,
)
async def get_model(
    model_id: Annotated[
        str,
        Path(description="Model ID", examples=["global.anthropic.claude-opus-5-5"]),
    ],
):
    await validate_model_id(model_id)
    return Model(id=model_id)
