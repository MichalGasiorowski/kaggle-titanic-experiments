#!/usr/bin/env bash

echo "publishing image ${LOCAL_IMAGE_NAME} to ECR..."

DEFAULT_REGION=$(aws configure get default.region)
AWS_REGION="${AWS_REGION:-$DEFAULT_REGION}"

DEFAULT_REGISTRY_ID=492542893717
REGISTRY_ID="${REGISTRY_ID:-$DEFAULT_REGISTRY_ID}"

DEFAULT_REPOSITORY_NAME=titanic-survival-lambda
REPOSITORY_NAME="${REPOSITORY_NAME:-$DEFAULT_REPOSITORY_NAME}"

# Create repository, ignore errors if it does already exist
aws ecr create-repository --repository-name "$REPOSITORY_NAME" || true

echo "Default region: $DEFAULT_REGION"
echo "Aws region: AWS_REGION"
echo "Local image name: $LOCAL_IMAGE_NAME"


AUTH_TOKEN=$(aws ecr get-login-password --region "$AWS_REGION")
echo "Auth token: $AUTH_TOKEN"

# Query ECR API and pipeline to docker login:

REPOSITORY_URI=$(aws ecr describe-repositories --registry-id "$REGISTRY_ID" --repository-names "$REPOSITORY_NAME" --query "repositories[].[repositoryUri][0][0]" | tr -d '"')
echo "REPOSITORY_URI : $REPOSITORY_URI"

docker login -u AWS -p "$AUTH_TOKEN" "$REPOSITORY_URI"

# Tag local docker image

docker tag "${LOCAL_IMAGE_NAME}" "$REPOSITORY_URI":latest

#Push to ECR:

docker push "$REPOSITORY_URI":latest