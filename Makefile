PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = mlflow-enkidupal-experiments
PROFILE = default
PROJECT_NAME = kaggle-titanic-experiments
PYTHON_INTERPRETER = python3
S3_BUCKET = s3://${BUCKET}
HPO_MAX_EVALS=100
HPO_MODEL='xgboost'

#LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_TAG:=v1
LOCAL_SERVICE_IMAGE_NAME:=titanic-experiment-predict-service:${LOCAL_TAG}
LOCAL_LAMBDA_IMAGE_NAME:=titanic-experiment-predict-lambda:${LOCAL_TAG}


pipenv_install:
	pipenv install

hpo: pipenv_install
	${PYTHON_INTERPRETER} -m src.models.hpo --max_evals ${HPO_MAX_EVALS} --model ${HPO_MODEL}

run_mlflow: pipenv_install
	mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts --artifacts-destination ${S3_BUCKET}

test:
	pytest src/tests/

quality_checks:
	pipenv install --dev
	isort .
	black .
	pylint --recursive=y .

build_service: quality_checks test
	docker build -f src/docker/predict/service/Dockerfile -t ${LOCAL_SERVICE_IMAGE_NAME} .

integration_test_service: build_service
	LOCAL_IMAGE_NAME=${LOCAL_SERVICE_IMAGE_NAME} bash src/integration-tests/run.sh

publish_service: integration_test_webservice
	LOCAL_IMAGE_NAME=${LOCAL_SERVICE_IMAGE_NAME} bash src/scripts/publish.sh

build_lambda: quality_checks test
	docker build -f src/docker/predict/serverless/Dockerfile -t ${LOCAL_LAMBDA_IMAGE_NAME} .

integration_test_lambda: build_lambda
	LOCAL_IMAGE_NAME=${LOCAL_LAMBDA_IMAGE_NAME} bash src/integration-tests/run.sh

publish_lambda: integration_test_lambda
	LOCAL_IMAGE_NAME=${LOCAL_LAMBDA_IMAGE_NAME} bash src/scripts/publish.sh

setup: pipenv_update
	pre-commit install