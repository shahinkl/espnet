#!/bin/bash
vols="-v ${PWD}/egs:/espnet/egs -v ${PWD}/espnet:/espnet/espnet -v ${PWD}/test:/espnet/test -v ${PWD}/utils:/espnet/utils"
ports="-p 6006:6006"
# shellcheck disable=SC2086
docker run $vols $ports -it espnet/espnet:cpu-u18 /bin/bash
