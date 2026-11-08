#!/usr/bin/env bash
# Build the Lambda image from the current commit, push it to ECR and deploy it
# through the CloudFormation stack, without prompts.
#
# The stack's ContainerImageUri is set to the pushed image digest, so the stack
# always records exactly which image runs (an unchanged ":latest" URI would not
# redeploy the function). Parameters that are not passed keep their current values.
#
# A build deploys only a commit that passed CI on GitHub (checked with the gh CLI),
# and every image gets a new tag (<commit>-<UTC time>). Uncommitted changes to the
# image source or the template are not covered by CI, so they need ALLOW_UNVERIFIED=1.
#
# The script sets the ECR repository's tags to immutable and a lifecycle policy that
# keeps the newest ECR_KEEP_DEPLOYED images that were ever deployed and the newest
# ECR_KEEP_IMAGES images overall. After the stack update completes it adds a
# "deployed-<UTC time>" tag to the image the function runs (also for IMAGE_URI
# rollbacks), and before a deploy it tags the image the stack is running, so
# --preview builds cannot expire the running image or the rollback targets.
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
#   ALLOW_DIRTY      Set to 1 to run from a working tree with uncommitted changes
#   IMAGE_URI        Deploy this image (repo@sha256:...) instead of building one. The
#                    template still comes from the working tree. Skips the CI check,
#                    but uncommitted template changes still need ALLOW_UNVERIFIED=1.
#   ALLOW_UNVERIFIED Set to 1 to deploy without a successful CI run for HEAD, or with
#                    uncommitted changes to src/ or the template
#   ECR_KEEP_DEPLOYED Deployed images the ECR lifecycle policy keeps (default: 10)
#   ECR_KEEP_IMAGES  Images the ECR lifecycle policy keeps in total, deployed or not
#                    (default: 10); deployed images beyond it are still kept
#   CI_REPO          GitHub repository (owner/name) whose CI runs are checked (default:
#                    from the "fork" remote, or "origin" if there is none)

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
keep_deployed="${ECR_KEEP_DEPLOYED:-10}"
keep_images="${ECR_KEEP_IMAGES:-10}"
if [[ ! "$keep_deployed" =~ ^[1-9][0-9]*$ || ! "$keep_images" =~ ^[1-9][0-9]*$ ]]; then
    echo "ECR_KEEP_DEPLOYED and ECR_KEEP_IMAGES must be positive integers." >&2
    exit 2
fi

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

template=deployment/BedrockProxy.template

# CI covers committed files only. A build deploys src/ and the template, an IMAGE_URI
# rollback the template, so uncommitted changes there are deployed unverified.
require_committed() {
    local paths=("$@")
    if [[ -z "$(git status --porcelain -- "${paths[@]}")" ]]; then
        return
    fi
    if [[ "${ALLOW_UNVERIFIED:-0}" != "1" ]]; then
        echo "Uncommitted changes in ${paths[*]} are not covered by CI; commit them or set ALLOW_UNVERIFIED=1." >&2
        exit 1
    fi
    echo "ALLOW_UNVERIFIED=1: deploying uncommitted changes in ${paths[*]}." >&2
}

# Refuses to continue unless the CI workflow succeeded for HEAD on GitHub.
require_ci_success() {
    local commit remote url ci_repo runs
    if [[ "${ALLOW_UNVERIFIED:-0}" == "1" ]]; then
        echo "ALLOW_UNVERIFIED=1: skipping the CI check." >&2
        return
    fi
    commit="$(git rev-parse HEAD)"
    ci_repo="${CI_REPO:-}"
    if [[ -z "$ci_repo" ]]; then
        remote=origin
        git remote get-url fork >/dev/null 2>&1 && remote=fork
        url="$(git remote get-url "$remote")"
        # https://github.com/owner/name(.git) or git@github.com:owner/name(.git)
        ci_repo="$(sed -E 's#^(https://|git@)github\.com[:/]##; s#\.git$##' <<< "$url")"
    fi
    if ! command -v gh >/dev/null; then
        echo "The gh CLI is needed to check CI for ${commit}; install it or set ALLOW_UNVERIFIED=1." >&2
        exit 1
    fi
    runs="$(gh run list --repo "$ci_repo" --commit "$commit" --workflow ci.yml --status success \
        --json databaseId --jq length)"
    if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
        echo "No successful CI run in ${ci_repo} for ${commit}; push it and wait for CI, or set ALLOW_UNVERIFIED=1." >&2
        exit 1
    fi
    echo "CI passed for ${commit} in ${ci_repo}"
}

# ECR lifecycle rules (https://docs.aws.amazon.com/AmazonECR/latest/userguide/LifecyclePolicies.html
# and lifecycle_policy_parameters.html): rules apply in rulePriority order, lowest first, and a
# "tagStatus": "any" rule must have the highest priority; "an image that matches the tagging
# requirements of a rule cannot be expired by a rule with a lower priority", so rule 2 never expires
# a deployed-* image. imageCountMoreThan keeps the newest images by push time and expires the rest;
# rule 2 still counts the deployed images among its newest ECR_KEEP_IMAGES. tagPatternList (a
# wildcard match, the recommended form) is used instead of tagPrefixList; a rule may have only one.
lifecycle_policy="$(printf '{"rules":[%s,%s]}' \
    "{\"rulePriority\":1,\"description\":\"Keep the newest ${keep_deployed} deployed images\",\"selection\":{\"tagStatus\":\"tagged\",\"tagPatternList\":[\"deployed-*\"],\"countType\":\"imageCountMoreThan\",\"countNumber\":${keep_deployed}},\"action\":{\"type\":\"expire\"}}" \
    "{\"rulePriority\":2,\"description\":\"Keep the newest ${keep_images} images\",\"selection\":{\"tagStatus\":\"any\",\"countType\":\"imageCountMoreThan\",\"countNumber\":${keep_images}},\"action\":{\"type\":\"expire\"}}")"
# The policy earlier versions of this script set, replaced without a notice.
previous_policy='{"rules":[{"rulePriority":1,"description":"Keep the newest 10 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}'

# Immutable tags and the lifecycle policy above. Both are idempotent and run on every deploy;
# a policy set by someone else is replaced with a notice that prints it.
configure_repository() {
    local current
    aws ecr put-image-tag-mutability \
        --region "$AWS_REGION" \
        --repository-name "$repository" \
        --image-tag-mutability IMMUTABLE >/dev/null
    if ! current="$(aws ecr get-lifecycle-policy \
        --region "$AWS_REGION" \
        --repository-name "$repository" \
        --query lifecyclePolicyText \
        --output text 2>&1)"; then
        if [[ "$current" != *LifecyclePolicyNotFoundException* ]]; then
            echo "$current" >&2
            exit 1
        fi
        current=""
    fi
    # ECR may reformat the stored JSON, so compare without whitespace.
    if [[ "$(tr -d '[:space:]' <<< "$current")" == "$(tr -d '[:space:]' <<< "$lifecycle_policy")" ]]; then
        return
    fi
    if [[ -n "$current" && "$(tr -d '[:space:]' <<< "$current")" != "$(tr -d '[:space:]' <<< "$previous_policy")" ]]; then
        echo "NOTICE: replacing the lifecycle policy of ECR repository ${repository}, which this script did not set:" >&2
        echo "$current" >&2
    fi
    aws ecr put-lifecycle-policy \
        --region "$AWS_REGION" \
        --repository-name "$repository" \
        --lifecycle-policy-text "$lifecycle_policy" \
        >/dev/null
    echo "ECR lifecycle policy: keep the newest ${keep_deployed} deployed images and ${keep_images} images in total"
}

# Adds a deployed-<UTC time> tag to the image $1 (<account>.dkr.ecr.<region>.amazonaws.com/<repo>@sha256:...)
# by putting its manifest again under the new tag, which immutable tags allow. --image-digest makes ECR
# reject the call if the manifest would get a different digest.
# https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-retag.html
tag_deployed() {
    local uri="$1" registry name digest account region manifest media_type tag output
    registry="${uri%%/*}"
    name="${uri#*/}"
    name="${name%@*}"
    digest="${uri#*@}"
    if [[ ! "$registry" =~ ^([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$ ]]; then
        echo "NOTICE: ${uri} is not in a private ECR registry; not tagging it as deployed." >&2
        return
    fi
    account="${BASH_REMATCH[1]}"
    region="${BASH_REMATCH[2]}"
    manifest="$(aws ecr batch-get-image --region "$region" --registry-id "$account" --repository-name "$name" \
        --image-ids "imageDigest=${digest}" --query 'images[0].imageManifest' --output text)"
    media_type="$(aws ecr batch-get-image --region "$region" --registry-id "$account" --repository-name "$name" \
        --image-ids "imageDigest=${digest}" --query 'images[0].imageManifestMediaType' --output text)"
    if [[ -z "$manifest" || "$manifest" == None ]]; then
        echo "Image ${uri} was not found in ECR." >&2
        exit 1
    fi
    tag="deployed-$(date -u +%Y%m%d%H%M%S)"
    # ImageAlreadyExistsException: the image already has this tag (tagged twice in one second).
    if ! output="$(aws ecr put-image --region "$region" --registry-id "$account" --repository-name "$name" \
        --image-tag "$tag" --image-digest "$digest" \
        --image-manifest "$manifest" --image-manifest-media-type "$media_type" 2>&1 >/dev/null)" \
        && [[ "$output" != *ImageAlreadyExistsException* ]]; then
        echo "$output" >&2
        echo "Could not tag ${uri} as deployed; the ECR lifecycle policy may expire it." >&2
        exit 1
    fi
    echo "Tagged ${name}@${digest} as ${tag}"
    if [[ "$region" == "$AWS_REGION" && "$name" == "$repository" ]]; then
        warn_if_unprotected "$digest"
    fi
}

# The lifecycle policy ranks deployed images by push time, so an old image deployed again (a rollback)
# can fall outside the newest ECR_KEEP_DEPLOYED.
warn_if_unprotected() {
    local digest="$1" newest
    newest="$(aws ecr describe-images --region "$AWS_REGION" --repository-name "$repository" \
        --filter tagStatus=TAGGED \
        --query "reverse(sort_by(imageDetails[?length(imageTags[?starts_with(@, 'deployed-')]) > \`0\`], &imagePushedAt))[:${keep_deployed}].imageDigest" \
        --output text)"
    if [[ "$newest" != *"$digest"* ]]; then
        echo "WARNING: ${digest} was pushed before the newest ${keep_deployed} deployed images, so the ECR lifecycle" >&2
        echo "policy will expire it within a day. Raise ECR_KEEP_DEPLOYED and run the script again to keep it." >&2
    fi
}

# Tags the image the stack runs now, if it has no deployed-* tag yet (it was deployed before this
# script tagged deployments), so it stays available as a rollback target.
tag_running_image() {
    local running tags
    running="$(aws cloudformation describe-stacks --region "$AWS_REGION" --stack-name "$stack_name" \
        --query "Stacks[0].Parameters[?ParameterKey=='ContainerImageUri'].ParameterValue" --output text 2>/dev/null)" || return 0
    if [[ "$running" != */"${repository}"@sha256:* ]]; then
        return
    fi
    tags="$(aws ecr describe-images --region "$AWS_REGION" --repository-name "$repository" \
        --image-ids "imageDigest=${running#*@}" --query 'imageDetails[0].imageTags' --output text 2>/dev/null)" || return 0
    if [[ "$tags" != *deployed-* ]]; then
        tag_deployed "$running"
    fi
}

# Sets image_uri. Not called through $(...), where bash would ignore errexit.
build_and_push() {
    local tag account_id registry image digest
    # A new tag on every build: tags are immutable, and rebuilding a commit picks up OS patches.
    tag="$(git rev-parse --short=12 HEAD)"
    [[ -n "$(git status --porcelain)" ]] && tag="${tag}-dirty"
    tag="${tag}-$(date -u +%Y%m%d%H%M%S)"

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

    configure_repository
    tag_running_image
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
    require_committed "$template"
    image_uri="$IMAGE_URI"
    configure_repository
    tag_running_image
else
    require_committed src "$template"
    require_ci_success
    build_and_push
fi

# PARAMETERS is a space-separated list of Key=Value pairs.
read -r -a extra_parameters <<< "${PARAMETERS:-}"
deploy_args=(
    --region "$AWS_REGION"
    --stack-name "$stack_name"
    --template-file "$template"
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
    exit 0
fi

# aws cloudformation deploy waits for the update and fails if it does not complete; check that the
# stack now runs this image before marking it deployed.
read -r stack_status deployed_uri <<< "$(aws cloudformation describe-stacks --region "$AWS_REGION" \
    --stack-name "$stack_name" \
    --query "[Stacks[0].StackStatus, Stacks[0].Parameters[?ParameterKey=='ContainerImageUri'].ParameterValue | [0]]" \
    --output text)"
# UPDATE_ROLLBACK_COMPLETE: an earlier update failed, and an empty change set left it so.
if [[ ! "$stack_status" =~ ^(CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE)$ \
    || "$deployed_uri" != "$image_uri" ]]; then
    echo "Stack ${stack_name} is ${stack_status} with ${deployed_uri}; not tagging ${image_uri} as deployed." >&2
    exit 1
fi
tag_deployed "$image_uri"
