#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
PROJECTDIR="$BASEDIR"/../../..
echo "$PROJECTDIR"

DOCKER_DIR="${BASEDIR}"/../../docker
echo "$DOCKER_DIR"
DOCKER_PREDICT_DIR="${DOCKER_DIR}"/predict
echo "$DOCKER_PREDICT_DIR"
DOCKER_FILE_DIR="${DOCKER_PREDICT_DIR}"/serverless
echo "$DOCKER_FILE_DIR"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=v1
    export LOCAL_IMAGE_NAME="titanic-experiment-predict-lambda:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -f DOCKER_FILE_DIR/Dockerfile  -t ${LOCAL_IMAGE_NAME} ../../..
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down

docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml up -d

sleep 5

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'single_lambda'

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'multi_lambda'

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 's3_path_lambda'


ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml logs
    docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down
    exit ${ERROR_CODE}
fi


docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down