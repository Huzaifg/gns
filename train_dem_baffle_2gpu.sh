#!/usr/bin/env zsh
#SBATCH --job-name=dem_train
#SBATCH -p gpu-a100-dev
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -A BCS20003
#SBATCH --time=2:00:00
#SBATCH -o dem_train_2gpu.out

ml cuda/12.0
ml cudnn
ml nccl

module load intel/19.1.1
module load impi/19.0.9
module load mvapich2-gdr/2.3.7
module load mvapich2/2.3.7

module load phdf5/1.10.4
module load python3/3.9.7
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

PARENT="/work/09943/huzaifg/ls6/"
source "${PARENT}/gns/venv/bin/activate"

# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)

n_gpu_per_node=2

mpiexec.hydra -np $NNODES -ppn 1 ./launch_helper.sh $n_gpu_per_node
#python3 -m gns.train mode="train" --config-path ./ --config-name config.yaml

PY_PID=$!
wait $PY_PID
echo "#"
