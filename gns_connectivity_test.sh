#!/bin/bash
#SBATCH --job-name=a100SmallBench
#SBATCH --time=48:0:0
#SBATCH -o Otest_%a.out
#SBATCH -e Otest_%a.err
#SBATCH -p gpu-a100-small
#SBATCH -A BCS20003
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH --array=1-10%10    # Launch 10 tasks, you can adjust concurrency with the % parameter

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

cd "${PARENT}/gns"
DATA_PATH="${PARENT}gns/benchmark/dataset/"
MODEL_PATH="${PARENT}gns/benchmark/models/"

# Calculate con_radius based on SLURM_ARRAY_TASK_ID, ranging from 0.0025 to 0.025
START_RADIUS=0.0025
END_RADIUS=0.025
STEP=$(echo "scale=4; ($END_RADIUS - $START_RADIUS) / 9" | bc)
RADIUS=$(echo "scale=4; $START_RADIUS + ($SLURM_ARRAY_TASK_ID - 1) * $STEP" | bc)

# Scale and format the con_radius for the output file name
RADIUS_FOR_FILENAME=$(echo "$RADIUS * 10000 / 1" | bc)

OUTPUT_FILE="bench_${RADIUS_FOR_FILENAME}.txt"

echo "CONNECTIVITY RADIUS: ${RADIUS}"
python3 -u -m gns.train --mode="rollout" --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="model-275000.pt" --train_state_file="train_state-275000.pt" --con_radius=$RADIUS >> "${PARENT}gns/benchmark/output/${OUTPUT_FILE}"
PY_PID=$!
wait $PY_PID
echo "#"
