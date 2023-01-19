#!/bin/sh

# Image name
export REGION=europe-west4-docker.pkg.dev
export REPO_NAME=map
export IMAGE_NAME=iris-learner
export IMAGE_TAG=IMAGE_TAG
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_URI=${REGION}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Clean old containers
sh ~/Docker/clean.sh

# Clean up artifacts from emacs.
rm ./*~

# Format the code as per PEP8.
{
    #autopep8 --ignore=E266 --in-place -r --aggressive .
    echo "Skipping autopep8 for now."
} ||
{
    echo "WARNING: pip3 install autopep8 if you would like to format your code."
}

# Build the Docker image.
docker build -f Dockerfile -t ${IMAGE_URI} ./
