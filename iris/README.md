
# Table of Contents

1.  [Serve](#org97c91a8)



<a id="org97c91a8"></a>

# Serve

The different ways to serve are as follows

-   `sh run.sh '[[6.3, 3.3, 6.0, 2.5], [5.4, 3.9, 1.7, 0.4], [0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]]' Yes`
-   `gcloud ai endpoints predict $ENDPOINT_ID --region=$REGION --json-request=$DATA_DIR_LOC/request.json`
-   `docker run --entrypoint /bin/bash -v $DATA_DIR_LOC:$DATA_DIR_REM --env INSIDE_GCP=$2 $IMAGE_URI -c "python3 learner.py serve '$1'"`
-   `docker compose run iris-serve` (under construction)

