#!/bin/sh

docker build . -t tc-temp
docker run --gpus=all -it --rm -u $(id -u):$(id -g) -v "$(pwd)/rl_algos/logs:/rl_algos/logs" tc-temp
