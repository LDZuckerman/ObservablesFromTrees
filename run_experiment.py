'''
Save exp.json file in parent directory of ObservablesFromTrees
In parent directory, run 'python ObservablesFromTrees/run_experiment.py -f exp.json -gpu 1'
'''

import os, json, shutil, argparse
import os.path as osp
from tkinter import FALSE
from networkx import hypercube_graph
from sympy import hyper
from dev.train_script import run

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)
parser.add_argument("-jobname", "--jobname", type=str, required=False)
args = parser.parse_args()

# Load construction dictionary from exp json file
exp_file = osp.expanduser(args.f) # osp.join(osp.abspath(''), args.f)
d = json.load(open(exp_file))
print(f"Experiment parameters: \n{d}\n")
print(f'\nSETTING UP EXPERIMENT\n') # print(f'\nParameters are the following -> \n{d}')
print(f"Loading experiment from {args.f}")

# Set up output and copy experiment json file to output dir 
out_pointer = osp.expanduser(f"{d['task_dir']}{d['exp_id']}") # Dir for saving all training outputs (if n_trial > 1, will create subfolders. Will also have subfolder for TF logging)
if not osp.exists(out_pointer):
    print(f"   Makeing output directory {out_pointer}/")
    os.makedirs(out_pointer)
print(f"   Copying experiment file into {out_pointer}/ for future reference")
shutil.copy(exp_file, out_pointer) 

# Run experiment (train and val for each of n_trials)
run(d['run_params'], d['data_params'], d['learn_params'], d['hyper_params'], out_pointer, d['exp_id'], args.gpu)

# Move slurm output file to output dir
if args.jobname is not None:
    shutil.move(f"SlurmOut/{args.jobname}", f"{out_pointer}/SlurmOut/")
    print(f"   Moved slurm output file to {out_pointer}/")
print('DONE')

