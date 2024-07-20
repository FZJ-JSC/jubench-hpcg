#!/usr/bin/env bash

HPCG_PATH=${1:-'hpcg-cuda'}

if [[ -f ${HPCG_PATH}/xhpcg && -f ${HPCG_PATH}/hpcg.sh ]]; then
  if [[ ! -f ${HPCG_PATH}/hpcg.bash ]]; then
    sed -e "s|^XHPCG=/workspace/hpcg-linux-x86_64/xhpcg|XHPCG=$( readlink -f ${HPCG_PATH}/xhpcg )|" ${HPCG_PATH}/hpcg.sh > ${HPCG_PATH}/hpcg.bash
    echo "hpcg.sh modified successfully and stored as hpcg.bash"
  else
    echo "hpcg.bash exists; ready for benchmarking"
  fi
else
  echo "please, place a copy of xhpcg and hpcg.sh from Nvidia\'s container image in src/hpcg-cuda/"
  exit 1
fi

