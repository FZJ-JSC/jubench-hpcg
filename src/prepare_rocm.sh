#!/usr/bin/env bash

BRANCHNAME=master
COMMITID=${2}
HPCG_PATH=${1:-'hpcg-rocm'}

if [[ ! -d ${HPCG_PATH} ]]; then
  git clone -b $BRANCHNAME --recursive https://github.com/ROCmSoftwarePlatform/rocHPCG.git ${HPCG_PATH} && cd ${HPCG_PATH} && git checkout $COMMITID
else
  echo "Directory rocHPCG already exists; assuming source code has been downloaded before"
  cd ${HPCG_PATH} && git checkout $COMMITID
fi

if [[ $? -eq 0 ]]; then
  echo "rocHPCG git repo is ready"
  exit 0
else
  echo "error preparing rocHPCG source code"
  exit 1
fi
