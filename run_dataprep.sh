#!/bin/bash -l

#SBATCH -J D%j                    # aka "job-name" (%j gives job id)
#SBATCH -p cca                    # aka "partition": "cca" instead of "cpu" allows you to use up to 40 nodes
#SBATCH -t 84:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -N 4                      # aka "nodes": number of nodes to request
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G per cpu-core is default) 
#SBATCH --output=SlurmOut/D%j.out # path relative to WORKING directory, NOT directory of this file

module purge
module load python
module load gcc
source ~/VENVS/torch_env/bin/activate

sizelim='None'
reslim=100

while getopts "n:v:s:r" flag; do
 case $flag in
   n) DS_name=$OPTARG;;
   v) tng_vol=$OPTARG;;
   s) sizelim=$OPTARG;;
   r) reslim=$OPTARG;;
 esac
done

srun python ObservablesFromTrees/run_dataprep.py -DS_name $DS_name -tng_vol $tng_vol -sizelim $sizelim -reslim $reslim

####
# In parent directory of ObservablesFromTrees, run 'sbatch ObservablesFromTrees/run_dataprep.sh -n DS3 -v 300 -s 20000'
####
