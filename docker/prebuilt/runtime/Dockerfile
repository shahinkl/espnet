FROM ubuntu:18.04
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG DOCKER_VER
ENV DOCKER_BUILT_VER ${DOCKER_VER}}

ARG NUM_BUILD_CORES=8
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

RUN DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && apt-get update -y && apt-get -y install --no-install-recommends \
        automake \
        autoconf \
        apt-utils \
        bc \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flac \
        gawk \
        gfortran \
        gnupg2 \
        git \
        libtool \
        python2.7 \
        python3 \
        sox \
        ffmpeg \
        subversion \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update

# Install Kaldi
RUN git clone https://github.com/kaldi-asr/kaldi

RUN cd /kaldi/tools && \
#    ./extras/install_mkl.sh -sp debian intel-mkl-64bit-2019.2-057 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    make all && \
    rm -r openfst-*/src && \
    ./extras/install_beamformit.sh && \
    ./extras/install_irstlm.sh && \
    cd /kaldi/src && \
    ./configure --shared --use-cuda=no && \
    make depend -j${NUM_BUILD_CORES} && \
    make -j${NUM_BUILD_CORES} && \
    find /kaldi/src -name "*.o" -exec rm -f {} \; && \
    find /kaldi/src -name "*.o" -exec rm -f {} \; 

WORKDIR /