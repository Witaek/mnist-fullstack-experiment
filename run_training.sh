#!/bin/bash

#Define the directories for mounting
SOURCE_DIR="$PWD/src"
MLFLOW_DIR="$PWD/mlruns"

#Run docker contrainer with mounted volumes

docker run -it \
    -v $SOURCE_DIR:/app/src \
    -v $MLFLOW_DIR:/mlruns \
    --name mnist-trainer \
    mnist_trainer \
    bash