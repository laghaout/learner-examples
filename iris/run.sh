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

# Import the environment variables.
. ./.env

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
