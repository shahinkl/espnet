#!/bin/bash

dwl_dir=$1
data_dir=$2

local/tedlium2/prepare_ted_data.sh "${dwl_dir}" "${data_dir}"
for dset in dev test train; do
  utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 "${data_dir}/${dset}.orig" "${data_dir}/${dset}"
done
