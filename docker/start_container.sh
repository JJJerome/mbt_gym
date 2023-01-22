#!/bin/bash
tag='latest'
# To add a single gpu, add the flag --gpus device=0, or to add all gpus add --gpus all
docker run --rm --gpus all --shm-size=10.24gb -v ${PWD}/../:/home/mbt_gym/ -p $1:$1 --name mbt_gym --user root -dit mbt_gym:$tag ./launcher.sh $1
