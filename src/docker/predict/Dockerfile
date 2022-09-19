FROM python:3.10.4-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "src/docker/predict/Pipfile", "src/docker/predict/Pipfile.lock", "setup.py", "./" ]

RUN pipenv install --system --deploy

COPY [ "src/", "/app/src" ]

RUN ls -la /app/*

EXPOSE 9696

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app:/app/src"

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "src.models.predict_rest:app" ]