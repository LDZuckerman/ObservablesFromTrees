#!/bin/bash -l

#SBATCH --j=pysr                 # aka "job-name": output file will be called '  '
#SBATCH -p gpu                   # aka "parition"
#SBATCH -t 32:59:00              # aka "time": maximum time for run hh:mm:ss
#SBATCH -C v100                  # aka "constraint": use nodes with only listed feature
#SBATCH -N 1                     # aka "nodes": number of nodes to request
#SBATCH --gpus=4                 # total GPUs to request 
#SBATCH --mem-per-cpu=12G        # memory per cpu-core (4G per cpu-core is default) 
#SBATCH --output=../SlurmOut/%j.out  

### Not sure how to use these correctly ###
##SBATCH --array=1-10             # Number of jobs to run in parallel [MAKES 10 OUT FILES (BUT ONLY IF cpus-per-task=1?)]
##SBATCH --gpus=4                 # total GPUs to request 
##SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded) [MAKES 4 OUT FILES]
##SBATCH --gres=gpu:1             # number of gpus per node [FAILS]


module purge
module load python
module load gcc
module load cuda
source ~/VENVS/torch_env/bin/activate

srun python run_experiment.py -f ../Task_samProps/exp1.json -gpu 1