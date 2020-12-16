#!/bin/sh

DOCKER_BUILDKIT=1 docker build . -t tc-temp

docker run --gpus=all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -u $(id -u):$(id -g) -v "$(pwd):/tc" tc-temp
