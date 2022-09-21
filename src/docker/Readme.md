#### To build service docker container & run it locally:

( build it in at project root level directory )

`docker build -f src/docker/predict/webservice/Dockerfile -t titanic-experiment-predict-service:v1 .`

`docker run -it --rm \
-p 9696:9696 \
-e RUN_ID="21cf301e4e7f459bae86218626159897" \
-e EXPERIMENT_NUMBER='1' \
-v /home/michal/.aws/credentials:/root/.aws/credentials:ro \
titanic-experiment-predict-service:v1`


To test service locally:
run from src/models :

`python test_predict.py --url 'http://localhost:9696/predict' --scenario 'single_service'`

or, using curl:

```
curl \ 
-X POST \ 
-H "Content-Type: application/json" \ 
-d  '[ { "Age": 45, "Sex": "female", "Pclass": 0, "Embarked": "C", "SibSp": 1, "Parch": 4, "Fare": 1111 } ]'
http://localhost:9696/predict
```


To build lambda image:

`docker build -f src/docker/predict/serverless/Dockerfile -t titanic-experiment-predict-lambda:v1 .`

```
docker run -it --rm \
-p 9000:8080 \
-e RUN_ID="21cf301e4e7f459bae86218626159897" \
-e EXPERIMENT_NUMBER='1' \
-v /home/michal/.aws/credentials:/root/.aws/credentials:ro \
titanic-experiment-predict-lambda:v1
```

To test lambda locally:
`python test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'single_lambda'`
or, using curl, from the terminal:

```
curl \
-X POST \
-H "Content-Type: application/json" \
-d '{"data": [ { "Age": 45, "Sex": "female", "Pclass": 0, "Embarked": "C", "SibSp": 1, "Parch": 4, "Fare": 1111 } ]}' \
http://localhost:9000/2015-03-31/functions/function/invocations
```

See more at:
https://aws.amazon.com/blogs/aws/new-for-aws-lambda-container-image-support/

#### To create AWS Elastic BeanStalk config :

( at src/docker/predict level, where Pipfile is located)
1) Install awsebcli : 
`pipenv install awsebcli --dev`
2) Init eb:
`eb init --source  -p docker -r eu-west-1 titanic-survival-serving`
3) To run locally, using Dockerfile:
`eb local run --port 9696`


#### To push image to dockerhub : 

`docker tag titanic-experiment-predict:v1  enkidupal/titanic-experiment-predict:v1`

`docker push enkidupal/titanic-experiment-predict:v1`


#### To setup ECR : 

eb init -p docker -r eu-north-1 titanic-survival-serving

`aws ecr create-repository --repository-name <repo_name> --region <region_name>`

`aws ecr create-repository --repository-name titanic-survival-serving --region eu-north-1`
Got URI:

492542893717.dkr.ecr.eu-north-1.amazonaws.com/titanic-survival-serving

Get authorization token :
`AUTH_TOKEN=$(aws ecr get-login-password --region eu-north-1)`

-> get encrypted token

#### Query ECR API and pipeline to docker login:

` aws ecr --region <region> | docker login -u AWS -p $AUTH_TOKEN <repo_uri>`


`aws ecr --region eu-north-1 | docker login -u AWS -p $AUTH_TOKEN 492542893717.dkr.ecr.eu-north-1.amazonaws.com/titanic-survival-serving`


#### Tag local docker image :

`docker tag <source_image_tag> <target_ecr_repo_uri>`

`docker tag titanic-experiment-predict-lambda:v1 492542893717.dkr.ecr.eu-north-1.amazonaws.com/titanic-survival-serving`

Push to ECR:

`docker push 492542893717.dkr.ecr.eu-north-1.amazonaws.com/titanic-survival-serving:latest`
