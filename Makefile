PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET_NAME ?= mlflow-enkidupal-experiments
PROFILE = default
PROJECT_NAME = kaggle-titanic-experiments
PYTHON_INTERPRETER = python3
S3_BUCKET = s3://${BUCKET_NAME}

HPO_MAX_EVALS ?= 100
HPO_MODEL ?= 'xgboost'

#LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_TAG:=v1
LOCAL_SERVICE_IMAGE_NAME:=titanic-experiment-predict-service:${LOCAL_TAG}
SERVICE_REPOSITORY_NAME=titanic-survival-service
LOCAL_LAMBDA_IMAGE_NAME:=titanic-experiment-predict-lambda:${LOCAL_TAG}
LAMBDA_REPOSITORY_NAME=titanic-survival-lambda


pipenv_install:
	pipenv install

# example usage:
# HPO_MAX_EVALS="30" HPO_MODEL="xgboost" make hpo
hpo: pipenv_install
	${PYTHON_INTERPRETER} -m src.models.hpo --max_evals ${HPO_MAX_EVALS} --model ${HPO_MODEL}

# example usage:
# BUCKET_NAME="MY_S3_BUCKET_NAME" make run_mlflow
run_mlflow: pipenv_install
	mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts --artifacts-destination ${S3_BUCKET}

test:
	pipenv run pytest src/tests/

quality_checks: pipenv_install
	pipenv run isort .
	pipenv run black .
	pipenv run pylint --recursive=y .

setup: pipenv_install
	pre-commit install

#service

service_build: quality_checks test
	docker build -f src/docker/predict/service/Dockerfile -t ${LOCAL_SERVICE_IMAGE_NAME} .

service_run: service_build
	docker-compose -f src/docker/predict/service/docker-compose.yaml up -d

service_down:
	docker-compose -f src/docker/predict/service/docker-compose.yaml down

service_integration_test: service_build
	LOCAL_IMAGE_NAME=${LOCAL_SERVICE_IMAGE_NAME} bash src/integration-tests/service/run.sh

service_publish: service_integration_test
	LOCAL_IMAGE_NAME=${LOCAL_SERVICE_IMAGE_NAME} REPOSITORY_NAME=${SERVICE_REPOSITORY_NAME}  bash src/scripts/publish.sh

# lambda image

lambda_build: quality_checks test
	docker build -f src/docker/predict/serverless/Dockerfile -t ${LOCAL_LAMBDA_IMAGE_NAME} .

lambda_run: lambda_build
	docker-compose -f src/docker/predict/serverless/docker-compose.yaml up -d

lambda_down:
	docker-compose -f src/docker/predict/serverless/docker-compose.yaml down

lambda_integration_test: lambda_build
	LOCAL_IMAGE_NAME=${LOCAL_LAMBDA_IMAGE_NAME} bash src/integration-tests/serverless/run.sh

lambda_publish: lambda_integration_test
	LOCAL_IMAGE_NAME=${LOCAL_LAMBDA_IMAGE_NAME} REPOSITORY_NAME=${LAMBDA_REPOSITORY_NAME} bash src/scripts/publish.sh

# lambda function

# example usage:
# RUN_ID="d7d3adc4407742d5ad9b7a695ebeb411" make lambda_function_create
lambda_function_create: #lambda_build
	RUN_ID=${RUN_ID} bash src/scripts/create_lambda.sh


#aws create-function --function-name titanic-survivorship-prediction --role lambda-kinesis-role 492542893717.dkr.ecr.eu-north-1.amazonaws.com/titanic-survival-lambda


# upload data to s3

upload_data:
	${PYTHON_INTERPRETER} -m src.data.write --path ${PROJECT_DIR}/data --bucketname ${BUCKET_NAME}