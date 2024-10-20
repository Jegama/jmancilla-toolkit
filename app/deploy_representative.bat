@REM gcloud config get-value project

@REM # set project
@REM gcloud config set project jgmancilla-agent-ltvjqg

@REM # build docker image
docker build -t jmancilla-toolkit .

@REM # tag docker image
docker tag jmancilla-toolkit us-west1-docker.pkg.dev/jgmancilla-agent-ltvjqg/jmancilla-toolkit/jmancilla-toolkit

@REM # create repository
@REM gcloud artifacts repositories create jmancilla-toolkit --repository-format=docker --location=us-west1 --description="jmancilla-toolkit"

@REM # push docker image
docker push us-west1-docker.pkg.dev/jgmancilla-agent-ltvjqg/jmancilla-toolkit/jmancilla-toolkit

@REM # deploy docker image to cloud run
gcloud run deploy jmancilla-toolkit --project jgmancilla-agent-ltvjqg --image us-west1-docker.pkg.dev/jgmancilla-agent-ltvjqg/jmancilla-toolkit/jmancilla-toolkit --region us-west1 --platform managed --allow-unauthenticated --port 80 --memory 512Mi --cpu 1 --timeout 300 --max-instances 4 --min-instances 1