#!/bin/sh

# Image name
export REGION=europe-west4-docker.pkg.dev
export REPO_NAME=map
export IMAGE_NAME=iris-learner
export IMAGE_TAG=IMAGE_TAG
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_URI=${REGION}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Data directory on the host.
DATA_DIR_LOC=/home/$(whoami)/Data/

# Data directory on the container.
DATA_DIR_REM=/home/Data/

# Map directory on the host.
MAP_DIR_LOC=$(pwd)/

# Map directory on the container.
MAP_DIR_REM=/home/

# Tidy up the directory.
sh clean.sh

# No arguments: Local-containerized without GCP flag.
if [ -z "$1" ]; then
    docker run -w $MAP_DIR_REM -v "$DATA_DIR_LOC":$DATA_DIR_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM --env INSIDE_GCP=No $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
# Single argument 0: Local-containerized without GCP flag, enter the bash.
elif [ $1 -eq 0 ]; then
    docker run  --entrypoint /bin/bash -w $MAP_DIR_REM -it -v "$MAP_DATA_LOC":$MAP_DATA_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM --env INSIDE_GCP=No $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
# Single argument 1: Local-containerized with GCP flag.
else
    docker run -w $MAP_DIR_REM -v "$DATA_DIR_LOC":$DATA_DIR_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
fi



