#!/bin/bash

join() { local IFS="$1"; shift; echo "$*"; }
echo_and_run() { echo "$@"; }

ps_ports=()
for ((i=0;i<ps_num;i++)); do
  ps_ports+=(0.0.0.0:$(($start_port+$i)))
done

worker_ports=()
for ((i=0;i<worker_num;i++)); do
  worker_ports+=(0.0.0.0:$(($start_port+ps_num+$i)))
done

ps_hosts=`join , "${ps_ports[@]}"`
worker_hosts=`join , "${worker_ports[@]}"`

for ((i=0;i<$ps_num;i++)); do
  echo_and_run CUDA_VISIBLE_DEVICES='' python main.py \
      --ps_hosts=$ps_hosts \
      --worker_hosts=$worker_hosts \
      --job_name=ps --task_index=$i "$@" \&
done

modular=$(($worker_num % $gpu_num))
if [ $modular == 0 ]; then
  add=0
  denom=$(($worker_num/$gpu_num))
else
  add=1
  denom=$(($worker_num/$gpu_num+1))
fi

for ((i=0;i<$worker_num;i++)); do
#  echo_and_run CUDA_VISIBLE_DEVICES=$((($i+$add)/$denom)) python main.py \
  echo_and_run CUDA_VISIBLE_DEVICES='' python main.py \
      --ps_hosts=$ps_hosts \
      --worker_hosts=$worker_hosts \
      --job_name=worker --task_index=$i "$@" \&
done
