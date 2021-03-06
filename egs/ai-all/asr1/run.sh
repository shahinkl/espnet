#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# general configuration
backend=pytorch

# start from -1 if you need to start from data download
stage=6
stop_stage=6

# number of gpus ("0" uses cpu, otherwise use gpu)
ngpu=8
nj=72
debugmode=1

# directory to dump full features
dumpdir=dump

# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
N=0

# verbose option
verbose=0

# Resume the training from snapshot
#resume="exp/train_pytorch_train/results/snapshot.ep.1"
resume=

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml

# current default recipe requires 4 gpus.
# if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
train_config=conf/train.yaml

lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
# specify a snapshot file to resume LM training
lm_resume=
# tag for managing LMs
lmtag=

# decoding parameter
# set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
recog_model=model.loss.best

# set a language model to be used for decoding
lang_model=rnnlm.model.best

# model average realted (only for transformer)
# the number of ASR models to be averaged
n_average=5

# if true, the validation `n_average`-best ASR models will be averaged.
# if false, the last `n_average` ASR models will be averaged.
use_valbest_average=true

# the number of languge models to be averaged
lm_n_average=0

# if true, the validation `lm_n_average`-best language models will be averaged.
# if false, the last `lm_n_average` language models will be averaged.
use_lm_valbest_average=false

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

# General configs
dwl_dir=downloads
data_dir=data
lang=en

# MAKE DIRECTORIES
mkdir -p "${dwl_dir}/ami"
mkdir -p "${dwl_dir}/commonvoice"
mkdir -p "${dwl_dir}/dipco"
mkdir -p "${dwl_dir}/librispeech"
mkdir -p "${dwl_dir}/tedlium2"
mkdir -p "${dwl_dir}/tedlium3"
mkdir -p "${dwl_dir}/voxforge"
mkdir -p "${data_dir}/ami"
mkdir -p "${data_dir}/commonvoice"
mkdir -p "${data_dir}/dipco"
mkdir -p "${data_dir}/librispeech"
mkdir -p "${data_dir}/tedlium2"
mkdir -p "${data_dir}/tedlium3"
mkdir -p "${data_dir}/voxforge"
mkdir -p "${data_dir}/combined/fbank"
mkdir -p "${data_dir}/combined/train"
mkdir -p "${data_dir}/combined/dev"
mkdir -p "${data_dir}/combined/test"
mkdir -p "${data_dir}/combined/train_org"
mkdir -p "${data_dir}/combined/dev_org"
mkdir -p "${data_dir}/combined/test_org"

# AMI
mic=ihm
# COMMONVOICE
data_url_cv=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/$lang.tar.gz
train_data_dir=train
dev_data_dir=dev
test_data_dir=test
# DIPCO
data_url_dc=https://s3.amazonaws.com/dipco/DiPCo.tgz
enhancement=beamformit
# LIBRISPEECH
data_url_ls=www.openslr.org/resources/12
# TEDLIUM 2
data_url_td2=http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz
# TEDLIUM 3
data_url_td3=http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz
data_type=legacy
# COMBINED
fbankdir="${data_dir}/combined/fbank"
train_set_org="${data_dir}/combined/train_org"
dev_set_org="${data_dir}/combined/dev_org"
train_set="${data_dir}/combined/train"
dev_set="${data_dir}/combined/dev"
test_set="${data_dir}/combined/test"

. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Starting stage -1: Data Download"
  # 1. AMI
  printf "\n\nStarting to download ami dataset ...\n"
  local/ami/download_and_arrange.sh "${mic}" "${dwl_dir}/ami" "${data_dir}/ami" &
  sleep 15
  # 2. COMMON VOICE
  printf "\n\nStarting to download common-voice dataset ...\n"
  local/commonvoice/download_and_untar.sh "${dwl_dir}/commonvoice" ${data_url_cv} ${lang}.tar.gz &
  sleep 15
  # 3. DIPCO
  printf "\n\nStarting to download dipco dataset ...\n"
  local/dipco/download_and_untar.sh "${dwl_dir}/dipco" ${data_url_dc} DiPCo.tgz &
  sleep 15
  # 4. LIBRISPEECH
  printf "\n\nStarting to download librispeech dataset ...\n"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/librispeech/download_and_untar.sh "${dwl_dir}/librispeech" ${data_url_ls} ${part} &
    sleep 90
  done
  #  # 5. TEDLIUM 2
  #  printf "\n\nStarting to download tedlium-2 dataset ...\n"
  #  local/tedlium2/download_and_untar.sh "${dwl_dir}/tedlium2" ${data_url_td2} TEDLIUM_release2.tar.gz &
  #  sleep 15
  # 6. TEDLIUM 3
  printf "\n\nStarting to download tedlium-3 dataset ...\n"
  local/tedlium3/download_and_untar.sh "${dwl_dir}/tedlium3" ${data_url_td3} TEDLIUM_release-3.tgz &
  sleep 15
  #  # 7. VOXFORGE
  #  printf "\n\nStarting to download voxforge dataset ...\n"
  #  local/voxforge/getdata.sh ${lang} "${dwl_dir}/voxforge" &
  wait # Wait for all process to complete
  printf "\n\n Completed stage -1: Data Download\n"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  ### Task dependent. You have to make data the following preparation part by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "Starting stage 0: Data preparation"
  # 1. AMI
  printf "\n\nStarting to prepare ami data ...\n"
  local/ami/prepare_data.sh "${mic}" "${dwl_dir}/ami" "${data_dir}/ami"

  # 2. COMMON VOICE
  printf "\n\nStarting to prepare common-voice data ...\n"
  local/commonvoice/prepare_data.sh "${dwl_dir}/commonvoice" "${data_dir}/commonvoice" "${lang}" "${train_data_dir}" "${dev_data_dir}" "${test_data_dir}"

  # 3. DIPCO
  printf "\n\nStarting to prepare dipco data ...\n"
  local/dipco/prepare_data.sh "${dwl_dir}/dipco" "${data_dir}/dipco" "$enhancement"

  # 4. LIBRISPEECH
  printf "\n\nStarting to prepare librispeech data ...\n"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/librispeech/data_prep.sh "${dwl_dir}/librispeech/LibriSpeech/${part}" "${data_dir}/librispeech/${part//-/_}"
  done

  #  # 5. TEDLIUM-2
  #  printf "\n\nStarting to prepare tedlium2 data ...\n"
  #  local/tedlium2/prepare_data.sh "${dwl_dir}/tedlium2" "${data_dir}/tedlium2"

  # 6. TEDLIUM-3
  printf "\n\nStarting to prepare tedlium3 data ...\n"
  local/tedlium3/prepare_data.sh "${dwl_dir}/tedlium3" "${data_dir}/tedlium3" "${data_type}"

  #  # 7. VOXFORGE
  #  printf "\n\nStarting to prepare voxforge data ...\n"
  #  local/voxforge/prepare_data.sh "${dwl_dir}/voxforge" "${data_dir}/voxforge" "${lang}"

  printf "\n\n Completed stage 0: Data preparation\n"
fi

#feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
#mkdir -p ${feat_tr_dir}
#feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
#mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  for x in "ami/ihm_train" \
    "commonvoice/train" \
    "dipco/dev_beamformit_ref" \
    "dipco/eval_beamformit_ref" \
    "librispeech/train_clean_100" \
    "librispeech/train_clean_360" \
    "librispeech/train_other_500" \
    "tedlium3/train" \
    "ami/ihm_dev" \
    "commonvoice/dev" \
    "dipco/dev_worn" \
    "librispeech/dev_clean" \
    "librispeech/dev_other" \
    "tedlium3/dev" \
    "ami/ihm_eval" \
    "commonvoice/test" \
    "dipco/eval_worn" \
    "librispeech/test_clean" \
    "librispeech/test_other" \
    "tedlium3/test"; do

    fbank_dir=${fbankdir}/$(echo ${x} | cut -d'/' -f1)
    mkdir -p "${fbank_dir}"
    printf "\n\nGenerating features for: %s\n" ${data_dir}/${x}
    steps/make_fbank.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true ${data_dir}/${x} exp/make_fbank/${x} ${fbank_dir}
    utils/fix_data_dir.sh ${data_dir}/${x}
  done

  utils/combine_data.sh --extra_files utt2num_frames ${train_set_org} "${data_dir}/ami/ihm_train" \
    "${data_dir}/commonvoice/train" \
    "${data_dir}/dipco/dev_beamformit_ref" \
    "${data_dir}/dipco/eval_beamformit_ref" \
    "${data_dir}/librispeech/train_clean_100" \
    "${data_dir}/librispeech/train_clean_360" \
    "${data_dir}/librispeech/train_other_500" \
    "${data_dir}/tedlium3/train"

  utils/combine_data.sh --extra_files utt2num_frames ${dev_set_org} "${data_dir}/ami/ihm_dev" \
    "${data_dir}/commonvoice/dev" \
    "${data_dir}/dipco/dev_worn" \
    "${data_dir}/librispeech/dev_clean" \
    "${data_dir}/librispeech/dev_other" \
    "${data_dir}/tedlium3/dev"

  utils/combine_data.sh --extra_files utt2num_frames ${test_set} "${data_dir}/ami/ihm_eval" \
    "${data_dir}/commonvoice/test" \
    "${data_dir}/dipco/eval_worn" \
    "${data_dir}/librispeech/test_clean" \
    "${data_dir}/librispeech/test_other" \
    "${data_dir}/tedlium3/test"

  # remove utt having more than 3000 frames
  # remove utt having more than 400 characters
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${train_set_org} ${train_set}
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${dev_set_org} ${dev_set}

#  # compute global CMVN
#  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
#
#  # dump features for training
#  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
#    utils/create_split_dir.pl \
#      /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
#      ${feat_tr_dir}/storage
#  fi
#  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
#    utils/create_split_dir.pl \
#      /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
#      ${feat_dt_dir}/storage
#  fi
#
#  dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
#    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
#  dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
#    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
#  for rtask in ${recog_set}; do
#    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#    mkdir -p ${feat_recog_dir}
#    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
#      data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
#      ${feat_recog_dir}
#  done
fi

dict="${data_dir}"/lang_char/train_${bpemode}${nbpe}_units.txt
bpemodel="${data_dir}"/lang_char/train_${bpemode}${nbpe}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p "${data_dir}"/lang_char/

  echo "<unk> 1" >${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
  cut -f 2- -d" " ${train_set}/text >${data_dir}/lang_char/input.txt
  spm_train --input=${data_dir}/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  spm_encode --model=${bpemodel}.model --output_format=piece <${data_dir}/lang_char/input.txt | tr ' ' '\n' | tr -d '[:punct:]' | tr -d '”' | tr -d '→' | tr -d '—' | tr -d 'به' | tr -d 'چی' | tr -d 'حرف' | tr -d 'راجع' | tr -d 'میزنی؟' | tr -d 'α' | tr -d 'π' | tr -d 'πgroup' | tr -d 'великий' | tr -d 'князь' | sort | uniq | awk '{print tolower($0) " " NR+1}' >>${dict}
  wc -l ${dict}

  # make json labels
  data2json.sh --nj ${nj} --feat ${train_set}/feats.scp --bpecode ${bpemodel}.model \
    ${train_set} ${dict} >${train_set}/data_${bpemode}${nbpe}.json

  data2json.sh --nj ${nj} --feat ${dev_set}/feats.scp --bpecode ${bpemodel}.model \
    ${dev_set} ${dict} >${dev_set}/data_${bpemode}${nbpe}.json

  data2json.sh --nj ${nj} --feat ${test_set}/feats.scp --bpecode ${bpemodel}.model \
    ${test_set} ${dict} >${test_set}/data_${bpemode}${nbpe}.json

#  for rtask in ${recog_set}; do
#    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#    data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
#      data/${rtask} ${dict} >${feat_recog_dir}/data_${bpemode}${nbpe}.json
#  done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
  lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: LM Preparation"
  lmdwl_dir=${data_dir}/lm/downloads
  mkdir -p ${lmdwl_dir}
  if [ ! -e ${lmdwl_dir}/librispeech-lm-norm.txt.gz ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${lmdwl_dir}
  fi

  if [ ! -e ${lmdwl_dir}/train_text.gz ]; then
    cut -f 2- -d" " ${train_set}/text | gzip -c >${lmdwl_dir}/train_text.gz
    cut -f 2- -d" " ${test_set}/text | gzip -c >${lmdwl_dir}/test_text.gz
  fi

  if [ ! -e ${data_dir}/lm/train.txt ]; then
    # combine external text and transcriptions and shuffle them with seed 777
    zcat ${lmdwl_dir}/librispeech-lm-norm.txt.gz ${lmdwl_dir}/train_text.gz ${lmdwl_dir}/test_text.gz |
      spm_encode --model=${bpemodel}.model --output_format=piece >${data_dir}/lm/train.txt
    cut -f 2- -d" " ${dev_set}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${data_dir}/lm/valid.txt
  fi

  ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
    lm_train.py \
    --config ${lm_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --verbose 1 \
    --outdir ${lmexpdir} \
    --tensorboard-dir tensorboard/${lmexpname} \
    --train-label ${data_dir}/lm/train.txt \
    --valid-label ${data_dir}/lm/valid.txt \
    --resume ${lm_resume} \
    --dict ${dict} \
    --dump-hdf5-path ${data_dir}/lm
fi

if [ -z ${tag} ]; then
  expname=train_${backend}_$(basename ${train_config%.*})
  if ${do_delta}; then
    expname=${expname}_delta
  fi
#  if [ -n "${preprocess_config}" ]; then
#    expname=${expname}_$(basename ${preprocess_config%.*})
#  fi
else
  expname=train_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Network Training"
  printf "\n\nTraining export directory: %s\n" "${expdir}"
  #  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
  #    asr_train.py \
  #    --config ${train_config} \
  #    --preprocess-conf ${preprocess_config} \
  #    --ngpu ${ngpu} \
  #    --backend ${backend} \
  #    --outdir ${expdir}/results \
  #    --tensorboard-dir tensorboard/${expname} \
  #    --debugmode ${debugmode} \
  #    --dict ${dict} \
  #    --debugdir ${expdir} \
  #    --minibatches ${N} \
  #    --verbose ${verbose} \
  #    --resume ${resume} \
  #    --train-json ${train_set}/data_${bpemode}${nbpe}.json \
  #    --valid-json ${dev_set}/data_${bpemode}${nbpe}.json

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    asr_train.py \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir tensorboard/${expname} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${train_set}/data_${bpemode}${nbpe}.json \
    --valid-json ${dev_set}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Decoding"
  if [[ $(get_yaml.py ${train_config} model-module) == *transformer* ]]; then
    # Average ASR models
    if ${use_valbest_average}; then
      recog_model=model.val${n_average}.avg.best
      opt="--log ${expdir}/results/log"
    else
      recog_model=model.last${n_average}.avg.best
      opt="--log"
    fi
    average_checkpoints.py \
      ${opt} \
      --backend ${backend} \
      --snapshots ${expdir}/results/snapshot.ep.* \
      --out ${expdir}/results/${recog_model} \
      --num ${n_average}

    # Average LM models
    if [ ${lm_n_average} -eq 0 ]; then
      lang_model=rnnlm.model.best
    else
      if ${use_lm_valbest_average}; then
        lang_model=rnnlm.val${lm_n_average}.avg.best
        opt="--log ${lmexpdir}/log"
      else
        lang_model=rnnlm.last${lm_n_average}.avg.best
        opt="--log"
      fi
      average_checkpoints.py \
        ${opt} \
        --backend ${backend} \
        --snapshots ${lmexpdir}/snapshot.ep.* \
        --out ${lmexpdir}/${lang_model} \
        --num ${lm_n_average}
    fi
  fi

  pids=() # initialize pids
  for rtask in ${recog_set}; do
    (
      decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
      feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

      # split data
      splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

      #### use CPU for decoding
      ngpu=0

      # set batchsize 0 to disable batch decoding
      ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${test_set}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model} \
        --rnnlm ${lmexpdir}/${lang_model} \
        --api v2

      score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
  done
  i=0
  for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
  [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
  echo "Finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: Tracing"
  ${trace_cmd} ${expdir}/train.log \
    asr_trace.py \
    --model_path "${expdir}/results/model.loss.best" \
    --lm_path "${expdir}/train_rnnlm_pytorch_lm_unigram5000_ngpu8/rnnlm.loss.best"
fi
