#!/bin/bash

mic=$1
dwl_dir=$2
data_dir=$3

base_mic=${mic//[0-9]/} # sdm, ihm or mdm

# common data prep
if [ ! -d "${data_dir}/local/downloads/annotations" ]; then
  local/ami/ami_text_prep.sh "${data_dir}/local/downloads/annotations"
fi
local/ami/"ami_${base_mic}_data_prep.sh" "${dwl_dir}" "${mic}"

local/ami/ami_"${base_mic}_scoring_data_prep.sh" "${dwl_dir}" "${mic}" dev
local/ami/ami_"${base_mic}_scoring_data_prep.sh" "${dwl_dir}" "${mic}" eval
for dset in train dev eval; do
  # changed the original AMI data structure in the Kaldi recipe to the following
  utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 "${data_dir}/${mic}/${dset}_orig data/${mic}_${dset}"
done
