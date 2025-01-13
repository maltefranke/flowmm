#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#SBATCH --time=07-00:00:00
####SBATCH --exclusive
#SBATCH --account=rwth1232


source ~/.bashrc

mamba activate flowmm

cd ~/flowmm/

export PYTHONPATH="${PYTHONPATH}:~/flowmm"

## Below will be set in the run_all_experiments.sh script, uncomment to run this script individually
#export sampling="uniform"
#export ot="true"

#export data="dual_docking_data_distributed"
#export model="docking_only_coords"
#export vectorfield="dual_docking_cspnet"

python scripts_model/run_docking.py data="$data" model="$model" vectorfield="$vectorfield" train.sampling="$sampling" data.datamodule.do_ot="$ot"