#!/bin/bash

dwl_dir=$1
data_dir=$2
lang=$3

selected=${dwl_dir}/${lang}/extracted
local/voxforge/voxforge_data_prep.sh "${selected}" "${data_dir}" "${lang}"
local/voxforge/voxforge_format_data.sh "${lang}" "${data_dir}"
