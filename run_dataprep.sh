#!/bin/bash -l

#SBATCH -J D%j                    # aka "job-name" (%j gives job id)
#SBATCH -p cca                    # aka "partition": "cca" instead of "cpu" allows you to use up to 40 nodes
#SBATCH -t 84:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -N 1                      # aka "nodes": number of nodes to request # DOES MULTITHREADING ALLOW ENOUGH SPEED UP THAT I CAN I FINALLY CHANGE THIS TO 1 (FOR BETTER OUTPUT)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G per cpu-core is default) 
#SBATCH --output=SlurmOut/D%j.out # path relative to WORKING directory, NOT directory of this file

module purge
module load python
module load gcc
source ~/VENVS/torch_env2/bin/activate

sizelim='None'
reslim=100

while getopts "n:v:s:r:d:m:" flag; do
 case $flag in
   n) DS_name=$OPTARG;;
   v) tng_vol=$OPTARG;;
   s) sizelim=$OPTARG;;
   r) reslim=$OPTARG;;
   d) downsize_method=$OPTARG;; 
   m) multi=$OPTARG;;
 esac
done

srun python ObservablesFromTrees/run_dataprep.py -DS_name $DS_name -tng_vol $tng_vol -sizelim $sizelim -reslim $reslim -downsize_method $downsize_method -multi $multi

####
# In parent directory of ObservablesFromTrees, run 'sbatch ObservablesFromTrees/run_dataprep.sh -n DS2 -v 100 -s None -r 100 -d 4 -m True'
####
