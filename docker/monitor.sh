#!/bin/bash

egs=$1
event_dir=$2
cmd="tensorboard --logdir=/tmp/egs/$egs/asr1/tensorboard/$event_dir --host=0.0.0.0"
docker run --name tensorboard -v /store/workspace/saal/espnet:/tmp -p 6006:6006 -i --rm tensorboad /bin/bash -c "$cmd"
