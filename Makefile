PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = mlflow-enkidupal-experiments
PROFILE = default
PROJECT_NAME = kaggle-titanic-experiments
PYTHON_INTERPRETER = python3


#LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_TAG:=v1
LOCAL_IMAGE_NAME:=titanic-experiment-predict-lambda:${LOCAL_TAG}

test:
	pytest src/tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integraton-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash src/scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install