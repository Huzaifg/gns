#!/bin/bash

#SBATCH -J pyt_train         # Job name
#SBATCH -o pyt_train.o%j     # Name of stdout output file
#SBATCH -e pyt_train.e%j     # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1                 # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH -A OTH21021          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
data="Sand-3D"
python3 -m gns.train --data_path="${SCRATCH}/gns_tensorflow/${data}/dataset" \
--model_path="${SCRATCH}/gns_tensorflow/${data}/models/" \
--output_path="${SCRATCH}/gns_tensorflow/${data}/rollouts/"
#--model_file="model-5000.pt" \
#--train_state_file="train_state-5000.pt"
