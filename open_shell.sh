#!/bin/bash
vols="-v ${PWD}/egs:/espnet/egs -v ${PWD}/espnet:/espnet/espnet -v ${PWD}/test:/espnet/test -v ${PWD}/utils:/espnet/utils"
# shellcheck disable=SC2086
docker run $vols -it espnet/espnet:gpu-cuda10.1-cudnn7-u18 /bin/bash
