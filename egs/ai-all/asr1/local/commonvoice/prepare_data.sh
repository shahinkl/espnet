#!/bin/bash

dwl_dir=$1
data_dir=$2
lang=$3
train_set=$4
train_dev=$5
test_set=$6

for part in "validated"; do
  # use underscore-separated names in data directories.
  local/commonvoice/data_prep.pl "${dwl_dir}" ${part} "${data_dir}"/"$(echo "${part}_${lang}" | tr - _)"
done
local/commonvoice/split_tr_dt_et.sh "${data_dir}/validated_${lang}" "${data_dir}/${train_set}" "${data_dir}/${train_dev}" "${data_dir}/${test_set}"
