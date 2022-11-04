FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "src/docker/predict/service/Pipfile", "src/docker/predict/service/Pipfile.lock", "setup.py", "./" ]

RUN pipenv install --system --deploy

COPY [ "src/", "/app/src" ]

EXPOSE 9696

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app:/app/src"

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "src.models.predict_rest:app" ]