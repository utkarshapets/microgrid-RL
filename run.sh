#!/bin/sh

docker build . -t tc
docker run --gpus=all -it -u $(id -u):$(id -g) -v "$(pwd)/rl_algos/logs:/rl_algos/logs" tc
