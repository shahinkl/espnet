#!/bin/bash
vols="-v ${PWD}/egs:/espnet/egs -v ${PWD}/espnet:/espnet/espnet -v ${PWD}/test:/espnet/test -v ${PWD}/utils:/espnet/utils"
docker run "$vols" -it espnet/espnet:cpu-18 /bin/bash