#!/bin/sh

# Image name
export REGION=europe-west4-docker.pkg.dev
export REPO_NAME=map
export IMAGE_NAME=iris
export IMAGE_TAG=IMAGE_TAG
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_URI=${REGION}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Work directory
WORKDIR=/home/

# Data directory on the host.
DATA_DIR_LOC=$(pwd)/

# Data directory on the container.
DATA_DIR_REM=/home/Data/

# No arguments: Run python3 learner.py serve
if [ -z "$1" ]; then
    docker run --entrypoint /bin/bash -v $DATA_DIR_LOC:$DATA_DIR_REM $IMAGE_URI -c 'python3 learner.py serve'
# Single argument 0: Get inside the CLI of the docker container.
elif [ "$1" == '0' ]; then
    docker run -it --entrypoint /bin/bash -v $DATA_DIR_LOC:$DATA_DIR_REM $IMAGE_URI
# Single argument: Use it as the serving data, for example:
# sh run.sh '[[6.3, 3.3, 6.0, 2.5], [5.4, 3.9, 1.7, 0.4]]'
# {"instances": [[6.3, 3.3, 6.0, 2.5], [5.4, 3.9, 1.7, 0.4]]}    
else
    docker run --entrypoint /bin/bash -v $DATA_DIR_LOC:$DATA_DIR_REM $IMAGE_URI -c "python3 learner.py serve '$1'"
fi
