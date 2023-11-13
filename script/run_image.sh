#!/bin/bash

# Default arguments
PORT=$(( ${RANDOM} % 9000 + 1000 ))
IMAGE_NAME=openpcdet
COMMAND=bash
IMAGE_NAME_=${IMAGE_NAME//\//_}
IMAGE_NAME_=${IMAGE_NAME_//:/_}
GPUS_ARGS="--gpus all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keep-alive) KEEP_ALIVE="true" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ ${KEEP_ALIVE} == 'true' ]]
then
    DOCKER_ARGS="-tid --restart always"
else
    DOCKER_ARGS="-ti --rm"
fi

docker run --ipc=host --shm-size=8gb --pid=host \
        ${DOCKER_ARGS} \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v ${HOME}:${HOME} \
        ${GPUS_ARGS} \
        -p 2${PORT}:8080 \
        -p 3${PORT}:8097 \
        -p 4${PORT}:8888 \
        -p 5${PORT}:6006 \
        --name ${USER}.${IMAGE_NAME_}.${PORT} \
        ${IMAGE_NAME} ${COMMAND}