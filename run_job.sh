#!/bin/bash

running_experiment_name=$1
args_experiment="${@:2}"

docker restart deep-rehab-pile-container
docker exec deep-rehab-pile-container bash -c "python3 -u main.py $args_experiment > $running_experiment_name".out""
