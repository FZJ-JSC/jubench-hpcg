#!/usr/bin/env bash

BRANCHNAME=master
COMMITID=${2}
HPCG_PATH=${1:-'hpcg-nvidia'}

if [[ ! -d ${HPCG_PATH} ]]; then
  git clone -b $BRANCHNAME --recursive https://github.com/NVIDIA/nvidia-hpcg.git ${HPCG_PATH} && cd ${HPCG_PATH} && git checkout $COMMITID
else
  echo "Directory ${HPCG_PATH} already exists; assuming source code has been downloaded before"
  cd ${HPCG_PATH} && git checkout $COMMITID
fi

if [[ $? -eq 0 ]]; then
  echo "nvidia-hpcg git repo is ready"
  exit 0
else
  echo "error preparing nvidia-hpcg source code"
  exit 1
fi
