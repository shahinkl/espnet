#!/bin/bash
mic=$1
dwl_dir=$2
data_dir=$3

if [ -d "${dwl_dir}" ] && ! touch "${dwl_dir}/.foo" 2>/dev/null; then
  echo "Directory ${dwl_dir} seems to exist and not be owned by you."
  echo "Assuming the data does not need to be downloaded.  Please use --stage 0 or more."
elif [ -e "${data_dir}/local/downloads/wget_${mic}.sh" ]; then
  echo "${data_dir}/local/downloads/wget_$mic.sh already exists, better skip than re-download."
else
  local/ami/download.sh "${mic}" "${dwl_dir}/ami"
fi
