FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "src/docker/predict/serverless/Pipfile", "src/docker/predict/serverless/Pipfile.lock", "setup.py", "./" ]

RUN pipenv install --system --deploy

COPY [ "src/", "/app/src" ]

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app:/app/src"

CMD [ "src.models.lambda_function.lambda_handler" ]