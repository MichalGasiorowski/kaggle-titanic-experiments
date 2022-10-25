#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=v1
    export LOCAL_IMAGE_NAME="titanic-experiment-predict-lambda:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -f ../../docker/predict/serverless/Dockerfile  -t ${LOCAL_IMAGE_NAME} ../../..
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose -f ${BASEDIR}/docker-compose.yaml down

docker-compose -f ${BASEDIR}/docker-compose.yaml up -d

sleep 5

pipenv run python ${BASEDIR}/test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'single_lambda'

pipenv run python ${BASEDIR}/test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'multi_lambda'


ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose -f ${BASEDIR}/docker-compose.yaml logs
    docker-compose -f ${BASEDIR}/docker-compose.yaml down
    exit ${ERROR_CODE}
fi



docker-compose -f ${BASEDIR}/docker-compose.yaml down