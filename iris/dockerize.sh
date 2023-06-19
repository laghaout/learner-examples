#!/bin/sh

# Image name
export REGION=europe-west4-docker.pkg.dev
export REPO_NAME=map
export IMAGE_NAME=iris
export IMAGE_TAG=IMAGE_TAG
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_URI=${REGION}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Run tests.
pytest

# Clean up artifacts
./clean.sh

# Build the Docker image.
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Generate the distribution and install the package locally [OPTIONAL].
python3 -m pip install --upgrade build
pip3 uninstall -y $IMAGE_NAME
python3 -m build
python3 -m pip install --user .
