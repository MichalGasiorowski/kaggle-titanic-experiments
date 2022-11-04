#!/usr/bin/env bash

echo "Creating lambda function... "

DEFAULT_REGION=$(aws configure get default.region)
AWS_REGION="${AWS_REGION:-$DEFAULT_REGION}"

DEFAULT_REGISTRY_ID=492542893717
REGISTRY_ID="${REGISTRY_ID:-$DEFAULT_REGISTRY_ID}"

DEFAULT_REPOSITORY_NAME=titanic-survival-lambda
REPOSITORY_NAME="${REPOSITORY_NAME:-$DEFAULT_REPOSITORY_NAME}"

DEFAULT_FUNCTION_NAME=titanic-survivorship-prediction-v2
FUNCTION_NAME="${FUNCTION_NAME:-$DEFAULT_FUNCTION_NAME}"

echo "Default region: $DEFAULT_REGION"
echo "Aws region: AWS_REGION"
echo "Local image name: $LOCAL_IMAGE_NAME"


REPOSITORY_URI=$(aws ecr describe-repositories --registry-id "$REGISTRY_ID" --repository-names "$REPOSITORY_NAME" --query "repositories[].[repositoryUri][0][0]" | tr -d '"')
echo "REPOSITORY_URI : $REPOSITORY_URI"

TAG=$(aws ecr list-images --registry-id "$REGISTRY_ID" --repository-name "$REPOSITORY_NAME" --query "imageIds[].[imageTag][0][0]" | tr -d '"')
echo "TAG : $TAG"

IMAGE_URI="$REPOSITORY_URI":"$TAG"
echo "IMAGE_URI : $IMAGE_URI"

aws lambda create-function --region "$AWS_REGION" --function-name "$FUNCTION_NAME" \
	--timeout 45 \
	--memory-size 256 \
  --package-type Image  \
  --code ImageUri="$IMAGE_URI"   \
  --role arn:aws:iam::492542893717:role/lambda-kinesis-role
