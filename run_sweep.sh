#!/bin/bash -l

#SBATCH -J E%j                    # aka "job-name" (%j gives job id)
#SBATCH -p gpu                    # aka "partition"
#SBATCH -t 32:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -C v100                   # aka "constraint": use nodes with only listed feature 
#SBATCH -N 1                      # aka "nodes": number of nodes to request
#SBATCH --gpus=4                  # total GPUs to request 
#SBATCH --mem=160G                # total memory to request     
#SBATCH --output=SlurmOut/E%j.out # path relative to WORKING directory, NOT directory of this file


module purge
module load python
module load gcc
module load cuda
source ~/VENVS/torch_env2/bin/activate

redo='False'
resume='False'

while getopts "f:d:r:s" flag; do
 case $flag in
   f) expfile=$OPTARG;;
   d) difffile=$OPTARG;;
   r) redo=$OPTARG;;
   s) resume=$OPTARG;;
 esac
done

srun python ObservablesFromTrees/run_sweep.py -f $expfile -d $difffile -gpu 1 -redo $redo -resume $resume 

