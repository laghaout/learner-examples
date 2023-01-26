#!/bin/sh
# |-------+-----+-------------------------------|
# | $1    | $2  | description                   |
# |-------+-----+-------------------------------|
# | ∅     | ∅   | launch the learner locally    |
# | Yes   | ∅   | launch the learner in the GCP |
# | bash  | No  | container terminal locally    |
# | bash  | Yes | container terminal in the GCP |
# | [···] | No  | serve [···] locally           |
# | [···] | Yes | serve [...] in the GCP        |
# |-------+-----+-------------------------------|

# Image name
export REGION=europe-west4-docker.pkg.dev
export REPO_NAME=map
export IMAGE_NAME=iris
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

if [ -z "$1" ]; then
    echo "Launch the learning pipeline locally."
    docker run -w $MAP_DIR_REM -v "$DATA_DIR_LOC":$DATA_DIR_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM --env INSIDE_GCP=No $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
elif [ "$1" = 'Yes' ]; then
    echo "Launch the learning pipeline in the GCP."
    docker run -w $MAP_DIR_REM -v "$DATA_DIR_LOC":$DATA_DIR_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM --env INSIDE_GCP=$1 $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
elif [ "$1" = 'bash' ] || [ "$1" = 'sh' ]; then
    echo "Get inside the container shell."
    docker run  --entrypoint /bin/$1 -w $MAP_DIR_REM -it -v "$DATA_DIR_LOC":$DATA_DIR_REM -v "$MAP_DIR_LOC":$MAP_DIR_REM --env INSIDE_GCP=$2 $IMAGE_URI 2>&1 | tee "$MAP_DIR_LOC/log.txt"
else
    echo "Serve."
    # sh run.sh '[[6.3, 3.3, 6.0, 2.5], [5.4, 3.9, 1.7, 0.4], [0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]]' Yes
    # gcloud ai endpoints predict $ENDPOINT_ID --region=$REGION --json-request=$DATA_DIR_LOC/request.json
    docker run --entrypoint /bin/bash -v $DATA_DIR_LOC:$DATA_DIR_REM --env INSIDE_GCP=$2 $IMAGE_URI -c "python3 learner.py serve '$1'"
fi
