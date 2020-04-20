#!/bin/bash

mic=$1
dwl_dir=$2
data_dir=$3

base_mic=${mic//[0-9]/} # sdm, ihm or mdm
nmics=${mic//[a-z]/}    # e.g. 8 for mdm8

# common data prep
if [ ! -d "${data_dir}/local/downloads/annotations" ]; then
  local/ami/ami_text_prep.sh "${data_dir}/local/downloads/annotations"
fi
local/ami/"ami_${base_mic}_data_prep.sh" "${dwl_dir}" "${mic}"
