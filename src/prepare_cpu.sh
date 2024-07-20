#!/usr/bin/env bash

BRANCHNAME=master
COMMITID=e64982640f0aa83f851fe3e1405c61d9a6d7321c
HPCG_PATH=${1:-'hpcg-cpu'}

if [[ ! -d ${HPCG_PATH} ]]; then
  git clone -b $BRANCHNAME --recursive https://github.com/hpcg-benchmark/hpcg.git ${HPCG_PATH} && cd ${HPCG_PATH} && git checkout $COMMITID
else
  echo "Directory hpcg-cpu already exists; assuming source code has been downloaded before"
  cd ${HPCG_PATH} && git checkout $COMMITID
fi

if [[ $? -eq 0 ]]; then
  echo "HPCG git repo is ready"
  exit 0
else
  echo "error preparing HPCG source code"
  exit 1
fi
