#!/bin/bash -l

#SBATCH --j=pysr                 # aka "job-name": output file will be called '  '
#SBATCH -p gpu                   # aka "parition"
#SBATCH -t 32:59:00              # aka "time": maximum time for run hh:mm:ss
#SBATCH -C v100                  # aka "constraint": use nodes with only listed feature
#SBATCH -N 1                     # aka "nodes": number of nodes to request
#SBATCH --array=1-10             # Number of jobs to run in parallel [WILL THIS WORK??]
#SBATCH --gpus=4                 # total GPUs to request
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks) [WILL THIS WORK??]
#SBATCH --mem-per-cpu=12G        # memory per cpu-core (4G per cpu-core is default) [WILL THIS WORK??]
#SBATCH --gres=gpu:1             # number of gpus per node [WILL THIS WORK??]
#SBATCH --output=SlurmOut/%j.out  

module purge
module load python
module load gcc
module load cuda
source ~/VENVS/torch_env/bin/activate

cd ~/ceph/ObsFromTrees_2023
srun python Mangrove_mod/run_experiment.py -f ExperimentFiles/exp_sam_props_recreate -gpu 1