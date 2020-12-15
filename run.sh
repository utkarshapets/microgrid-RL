#!/bin/sh

docker build . -t tc-temp
if $1
then
  docker run --gpus=all -it --rm -u $(id -u):$(id -g) -v "$(pwd):/tc" tc-temp $1
else
  docker run --gpus=all -it --rm -u $(id -u):$(id -g) -v "$(pwd):/tc" tc-temp
fi
