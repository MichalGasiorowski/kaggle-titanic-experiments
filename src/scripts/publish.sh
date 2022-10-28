#!/usr/bin/env bash

echo "publishing image ${LOCAL_IMAGE_NAME} to ECR..."

DEFAULT_REGION=$(aws configure get default.region)
AWS_REGION="${AWS_REGION:-$DEFAULT_REGION}"

echo "Default region: $DEFAULT_REGION"
echo "Aws region: AWS_REGION"
echo "Local image name: $LOCAL_IMAGE_NAME"

AUTH_TOKEN=$(aws ecr get-login-password --region $AWS_REGION)

echo "Auth token: $AUTH_TOKEN"
