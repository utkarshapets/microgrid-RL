#!/bin/sh

DOCKER_BUILDKIT=1 docker build . -t tc-temp

docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v "$(pwd):/tc" tc-temp
