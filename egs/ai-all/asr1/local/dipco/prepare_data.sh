#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

dwl_dir=$1
data_dir=$2
enhancement=$3
mictype=worn

mkdir -p "${dwl_dir}/DiPCo/enhancement"

json_dir="${dwl_dir}/DiPCo/transcriptions"
audio_dir="${dwl_dir}/DiPCo/audio"
enhandir="${dwl_dir}/DiPCo/enhancement"

for dset in dev eval; do
  local/dipco/prepare_dipco_data.sh --mictype ${mictype} "${audio_dir}/${dset}" "${json_dir}/${dset}" "${data_dir}/${dset}_${mictype}"
done

for dset in dev eval; do
  for mictype in u01 u02 u03 u04 u05; do
    local/dipco/run_beamformit.sh --cmd "$train_cmd" --bmf "1 2 3 4 5 6 7" "${audio_dir}/${dset}" "${enhandir}/${dset}_${enhancement}_${mictype}" ${mictype} &
  done
  wait
done

for dset in dev eval; do
  # The ref mic is the same as the worn: close-talk
  for mictype in u01 u02 u03 u04 u05; do
    local/dipco/prepare_dipco_data.sh --mictype ${mictype} "${enhandir}/${dset}_${enhancement}_${mictype}" "${json_dir}/${dset}" "${data_dir}/${dset}_${enhancement}_ref_${mictype}"
  done

  ddirs=$(ls -d "${data_dir}/${dset}_${enhancement}"_ref_u0*)
  utils/combine_data.sh "${data_dir}/${dset}_${enhancement}_ref" "${ddirs}"
  rm -rf "${data_dir}/${dset}_${enhancement}"_ref_u0*
done
# only use left channel for worn mic recognition
# you can use both left and right channels for training
for dset in dev eval; do
  utils/copy_data_dir.sh "${data_dir}/${dset}_worn data/${dset}_worn_stereo"
  grep "\.L-" "${data_dir}/${dset}_worn_stereo/text" > "${data_dir}/${dset}_worn/text"
  utils/fix_data_dir.sh "${data_dir}/${dset}_worn"
done
