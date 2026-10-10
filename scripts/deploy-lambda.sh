#!/usr/bin/env bash
# Build the Lambda image from the current commit, push it to ECR and deploy it
# through the CloudFormation stack, without prompts.
#
# The stack's ContainerImageUri is set to the pushed image digest, so the stack
# always records exactly which image runs (an unchanged ":latest" URI would not
# redeploy the function). Parameters that are not passed keep their current values.
#
# Usage:
#   AWS_REGION=ap-northeast-2 scripts/deploy-lambda.sh             # build, push and deploy
#   AWS_REGION=ap-northeast-2 scripts/deploy-lambda.sh --preview   # stop at a change set
#   AWS_REGION=ap-northeast-2 IMAGE_URI=<repo>@sha256:<digest> scripts/deploy-lambda.sh
#                                                       # redeploy a pushed image, e.g. to roll back
#
# Environment:
#   AWS_REGION       Region of the stack and ECR repository (required)
#   STACK_NAME       CloudFormation stack name (default: BedrockAccessGateway)
#   ECR_REPOSITORY   ECR repository name (default: bedrock-proxy-api)
#   PARAMETERS       Extra "Key=Value" stack parameter overrides, space separated (optional)
#   ALLOW_DIRTY      Set to 1 to build from a working tree with uncommitted changes
#   IMAGE_URI        Deploy this image (repo@sha256:...) instead of building one. The
#                    template still comes from the working tree.

set -o errexit
set -o nounset
set -o pipefail

preview=false
if [[ "${1:-}" == "--preview" ]]; then
    preview=true
elif [[ $# -gt 0 ]]; then
    echo "Unknown argument: $1" >&2
    exit 2
fi

: "${AWS_REGION:?Set AWS_REGION to the stack Region}"
stack_name="${STACK_NAME:-BedrockAccessGateway}"
repository="${ECR_REPOSITORY:-bedrock-proxy-api}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "The working tree has uncommitted changes; commit them or set ALLOW_DIRTY=1." >&2
    exit 1
fi

if [[ "${PARAMETERS:-}" == *ContainerImageUri=* ]]; then
    echo "Set IMAGE_URI instead of passing ContainerImageUri in PARAMETERS." >&2
    exit 2
fi

# Sets image_uri. Not called through $(...), where bash would ignore errexit.
build_and_push() {
    local tag account_id registry image digest
    tag="$(git rev-parse --short=12 HEAD)"
    [[ -n "$(git status --porcelain)" ]] && tag="${tag}-dirty"

    account_id="$(aws sts get-caller-identity --query Account --output text)"
    registry="${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    image="${registry}/${repository}"
    echo "Account ${account_id}, Region ${AWS_REGION}, stack ${stack_name}, image ${image}:${tag}"

    # --provenance/--sbom off: Lambda rejects image indexes with attestations.
    docker buildx build \
        --platform linux/arm64 \
        --provenance=false \
        --sbom=false \
        --tag "${image}:${tag}" \
        --file src/Dockerfile \
        --load \
        src/

    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$registry" >/dev/null
    docker push "${image}:${tag}" >/dev/null

    digest="$(aws ecr describe-images \
        --region "$AWS_REGION" \
        --repository-name "$repository" \
        --image-ids "imageTag=${tag}" \
        --query 'imageDetails[0].imageDigest' \
        --output text)"
    image_uri="${image}@${digest}"
    echo "Pushed ${image_uri}"
}

if [[ -n "${IMAGE_URI:-}" ]]; then
    if [[ "$IMAGE_URI" != *@sha256:* ]]; then
        echo "IMAGE_URI must reference a digest (repo@sha256:...)." >&2
        exit 2
    fi
    image_uri="$IMAGE_URI"
else
    build_and_push
fi

# PARAMETERS is a space-separated list of Key=Value pairs.
read -r -a extra_parameters <<< "${PARAMETERS:-}"
deploy_args=(
    --region "$AWS_REGION"
    --stack-name "$stack_name"
    --template-file deployment/BedrockProxy.template
    --capabilities CAPABILITY_IAM
    --no-fail-on-empty-changeset
    --parameter-overrides "ContainerImageUri=${image_uri}" ${extra_parameters[@]+"${extra_parameters[@]}"}
)
if [[ "$preview" == true ]]; then
    deploy_args+=(--no-execute-changeset)
fi

aws cloudformation deploy "${deploy_args[@]}"

if [[ "$preview" == true ]]; then
    echo "Change set created but not executed. Review it in the CloudFormation console or with"
    echo "  aws cloudformation describe-change-set --region ${AWS_REGION} --stack-name ${stack_name} --change-set-name <name>"
fi
