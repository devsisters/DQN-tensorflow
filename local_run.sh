#!/bin/sh

cpu=True

python main.py \
     --ps_hosts=0.0.0.0:2222 \
     --worker_hosts=0.0.0.0:2223,0.0.0.0:2224 \
     --job_name=ps --task_index=0 &

python main.py \
     --ps_hosts=0.0.0.0:2222 \
     --worker_hosts=0.0.0.0:2223,0.0.0.0:2224 \
     --job_name=worker --task_index=0 --cpu=$cpu &

python main.py \
     --ps_hosts=0.0.0.0:2222 \
     --worker_hosts=0.0.0.0:2223,0.0.0.0:2224 \
     --job_name=worker --task_index=1 --cpu=$cpu &

python main.py \
     --ps_hosts=0.0.0.0:2222 \
     --worker_hosts=0.0.0.0:2223 \
     --job_name=worker --task_index=0
