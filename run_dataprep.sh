#!/bin/bash -l

#SBATCH -J D%j                    # aka "job-name" (%j gives job id)
#SBATCH -p cca                    # aka "partition": "cca" instead of "cpu" allows you to use up to 40 nodes
#SBATCH -t 84:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -N 1                      # aka "nodes": number of nodes to request # NO NEED TO HAVE THIS GREATER THAN 1, RIGHT?
#SBATCH --mem-per-cpu=15G         # memory per cpu-core (4G per cpu-core is default) 
#SBATCH --output=SlurmOut/D%j.out # path relative to WORKING directory

##SBATCH --ntasks-per-node=10     # this will literally run the srun command 10 times!!! CANNOT USE THIS - GIVES ERROR BECASUE TRIES TO CREAT OUT DIR MULTIPLE TIMES! # Do I need this with multithreading? E.g. should this be 10 when I set n_thread = 10?


module purge
module load python
module load gcc
source ~/VENVS/torch_env2/bin/activate

sizelim='None'
reslim=100

while getopts "n:v:s:r:d:m:p:" flag; do

 case $flag in
   n) DS_name=$OPTARG;;
   v) tng_vol=$OPTARG;;
   s) sizelim=$OPTARG;;
   d) downsize_method=$OPTARG;; 
   m) multi=$OPTARG;;
   p) prep_props=$OPTARG;;
 esac
done

srun python ObservablesFromTrees/run_dataprep.py -DS_name $DS_name -tng_vol $tng_vol -sizelim $sizelim -downsize_method $downsize_method -multi $multi -prep_props $prep_props


####
# In parent directory of ObservablesFromTrees, run 'sbatch ObservablesFromTrees/run_dataprep.sh -n DS2 -v 100 -s None -r 100 -d 4 -m True'
####
