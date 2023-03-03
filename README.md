kaggle-titanic-experiments
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` `make hpo`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile            <- Pipfile for main project reproducting enviornment, e.g.
    │                         generated with `pipenv update`
    ├── Pipfile.lock       <- exact Pipfile with exact packages version 
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to download or generate data
    │   │   ├── download.py <- class and methods to download / save data
    │   │   └── read.py     <- class and methods to read the data 
    │   │
    │   └── docker                  <- Dockerfiles and scripts to build docker images  
    │   │   └── predict             <- docker image building inference ( prediction )
    │   │      ├── serverles        <- docker image building for serverless docker image ( lambda )
    │   │      └── service          <- docker image building for webservice docker image ( HTTP server - gunicorn )
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                   predictions
    │   │   ├── hpo.py              <- hyperparameter tuning script
    │   │   ├── train_model.py      <- train the model script
    │   │   ├── predict.py          <- load the model and make predictction 
    │   │   ├── lambda_function.py  <- lambda_handler function definition  
    │   │   └── predict_rest.py     <- HTTP REST endpoint for predict definition  
    │   │
    │   ├── scripts                             <- Scripts to publish images to AWS ECR
    │   ├── tests                               <- tests
    │   ├── integration-tests                   <- integration tests
    │   │   ├── serverless                      <- integration tests for serverless lambda-based inference
    │   │   │   ├── docker-compose.yaml   <- docker-compose file for creating docker image for serverless integration testing
    │   │   │   ├── run.sh                <- script to run integration test scenario
    │   │   │   └── test_predict.py       <- python script for testing 
    │   │   └── service                         <- integration tests for HTTP API based service
    │   │   │   ├── docker-compose.yaml   <- docker-compose file for creating docker image for service integration testing
    │   │   │   ├── run.sh                <- script to run integration test scenario
    │   │   │   └── test_predict.py       <- python script for testing inference service
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tox.ini                   <- tox file with settings for running tox; see tox.readthedocs.io
    └── pyproject.toml            <- pyproject.toml file with unified Python project settings file


--------

<p><small>Template based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

To run mlflow server:

`BUCKET_NAME="MY_S3_BUCKET_NAME" make run_mlflow` 

( specify bucket name ex: 'mlflow-enkidupal-experiments' )

`mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts --artifacts-destination s3://mlflow-enkidupal-experiments/`

To run hyperparameter tuning:

`HPO_MAX_EVALS="30" HPO_MODEL="xgboost" make hpo`


