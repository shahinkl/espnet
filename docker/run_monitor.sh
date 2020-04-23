#!/bin/bash

egs=$1
event_dir=$2
cmd="tensorboard --logdir=/tmp/egs/$egs/asr1/tensorboard/$event_dir --host=0.0.0.0"
docker_image=$(docker images -q tensorboad:latest)
if ! [[ -n ${docker_image} ]]; then
  docker build -f prebuilt/monitor/Dockerfile -t tensorboad:latest
fi
docker run --name tensorboard -v /store/workspace/saal/espnet:/tmp -p 6006:6006 -i --rm tensorboad /bin/bash -c "$cmd"
