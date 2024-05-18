#!/bin/bash -l

#SBATCH -J E%j                    # aka "job-name" (%j gives job id)
#SBATCH -p gpu                    # aka "partition"
#SBATCH -t 32:59:00               # aka "time": maximum time for run hh:mm:ss
#SBATCH -C v100                   # aka "constraint": use nodes with only listed feature (I think this is the only type on Popeye though?)
#SBATCH -N 1                      # aka "nodes": number of nodes to request
#SBATCH --gpus=4                  # total GPUs to request 
#SBATCH --mem=160G                # total memory to request (will 160G resolve the out-of-memory error?) [putting this in in place of #SBATCH --mem-per-cpu=24G     # memory per cpu-core (4G per cpu-core is default)]
#SBATCH --output=SlurmOut/E%j.out # path relative to WORKING directory, NOT directory of this file

### Not sure how to use these correctly ###
##SBATCH --array=1-10             # Number of jobs to run in parallel [MAKES 10 OUT FILES (BUT ONLY IF cpus-per-task=1?)]
##SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded) [MAKES 4 OUT FILES]
##SBATCH --gres=gpu:1             # number of gpus per node [FAILS]

#module purge
module load python
module load gcc
module load cuda
source ~/VENVS/torch_env/bin/activate

redo='False'

while getopts "f:r" flag; do
 case $flag in
   f) expfile=$OPTARG;;
   r) redo=$OPTARG;;
 esac
done

redo='True'

echo "Running experiment with expfile: $expfile"

srun python ObservablesFromTrees/run_experiment_dummy.py 

#srun python ObservablesFromTrees/run_experiment.py  -gpu 1 -f $expfile -redo $redo  # -jobname $SLURM_JOB_ID Really shouldn't pass job id to the script, but it's a quick fix for now.


####
# Save expfile.json file in ObservablesFromTrees/../Task_phot/ (or whatever the task is)
# In parent directory, run 'sbatch ObservablesFromTrees/run_exp.sh -f Task_phot/expfile.json' (or whatever the task and expfile is)
####