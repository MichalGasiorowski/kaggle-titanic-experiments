#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
PROJECTDIR="$BASEDIR"/../../..
echo "$PROJECTDIR"

DOCKER_DIR="${BASEDIR}"/../../docker
echo "$DOCKER_DIR"
DOCKER_PREDICT_DIR="${DOCKER_DIR}"/predict
echo "$DOCKER_PREDICT_DIR"
DOCKER_FILE_DIR="${DOCKER_PREDICT_DIR}"/service
echo "$DOCKER_FILE_DIR"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=v1
    export LOCAL_IMAGE_NAME="titanic-experiment-predict-service:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -f DOCKER_FILE_DIR/Dockerfile  -t ${LOCAL_IMAGE_NAME} ../../..
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down

docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml up -d

sleep 5

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9696/predict' --scenario 'single_service'

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9696/predict' --scenario 'multi_service'

pipenv run python "${DOCKER_PREDICT_DIR}"/test_predict.py --url 'http://localhost:9696/predict_from_s3_path' --scenario 's3_path_service'


ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml logs
    docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down
    exit ${ERROR_CODE}
fi


docker-compose -f "${DOCKER_FILE_DIR}"/docker-compose.yaml down