#!/bin/bash
egs=cv_trans
event_dir=valid_train_en_pytorch_train
cmd="tensorboard --logdir=/tmp/egs/$egs/asr1/tensorboard/$event_dir --host=0.0.0.0"
docker run --name tensorboard -v /store/workspace/saal/espnet:/tmp -p 6006:6006 -i --rm tensorboad /bin/bash -c "$cmd"
