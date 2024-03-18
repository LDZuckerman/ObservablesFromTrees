#!/bin/bash -l

#SBATCH --j=pysr                  # aka "job-name": as set will call it just 'xxxxxxxx.out'
#SBATCH -p cca                    # aka "partition": "cca" instead of "cpu" allows you to use up to 40 nodes
#SBATCH -t 42:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -N 4                      # aka "nodes": number of nodes to request
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G per cpu-core is default) 
#SBATCH --output=SlurmOut/%j.out  # path relative to WORKING directory, NOT directory of this file

module purge
module load python
module load gcc
source ~/VENVS/torch_env/bin/activate

srun python run_dataprep.py -DS_name 'DS1' -downsize_method 3 -sizelim 20000

####
# In parent directory of ObservablesFromTrees, run 'sbatch ObservablesFromTrees/run_exp.sh'
####
