#!/bin/bash

LOCAL_RANK=$PMI_RANK

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


CMD="torchrun --nproc_per_node $1 --nnodes $NNODES --node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --master_port=1234 ./gns/train.py --config-path ./ --config-name config.yaml"

FULL_CMD="$CMD"
echo "Training command: $FULL_CMD"

eval $FULL_CMD
