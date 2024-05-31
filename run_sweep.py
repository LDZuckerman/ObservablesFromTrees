'''
Perform a hyperparameter search by running a large number of experiments.

Each experiment:
    Saves trained model to sweep directory.
    Adds dict of final validation results to results file (not test results - compute those separately for best models)
    Adds dict of experiment parameters to done experiments file.

Input files: 
    Difffile = dict containing lists of valyes to sweep through for parameters to explore.
    Basefile = experiment dict containing values to set for parameters held constant (not that base file itself is modified, so can be existing exp file)

NOTE: If another sweep is run on the same task, experiments will be added to the same output directory. 
      If multiple seperate sweeps are desired (e.g. two different datasets), could add sweep id to base file and modify below to create a new output directory for each sweep.
'''

import os, shutil, argparse
import json, pickle
import os.path as osp
import pandas as pd
import numpy as np
import dev.train_script as train_script
import dev.run_utils as run_utils
import multiprocessing as mp
import copy

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-d", "--d", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False, default='1')
parser.add_argument("-resume", "--resume", type=str, required=False, default='False')
parser.add_argument("-add", "--add", type=str, required=False, default='False')
args = parser.parse_args()

# Load base and diff dicts
basedict = json.load(open(osp.expanduser(args.f)))
diff = json.load(open(osp.expanduser(args.d)))
sweep_dict = diff['sweep_params']
sweep_id = diff['sweep_id']

# Get list of config dicts using base and diff files
diffsets = run_utils.perms(sweep_dict) 
diffsets = pd.DataFrame(data=diffsets, columns=sweep_dict.keys())
all_paramdicts = []
for i in range(len(diffsets)):
    dict = copy.deepcopy(basedict)
    diffset = diffsets.iloc[i]
    for groupkey in ['learn_params', 'hyper_params', 'run_params']: # assume keeping data params constant (perhaps do another sweep if want to change them)
        for key in basedict[groupkey].keys():
            if key in sweep_dict.keys():
                dict[groupkey][key] = diffset[key]
    expid = f"exp{sweep_id.replace('sweep','')}{'{:03d}'.format(i)}" # e.g. exp0001 for first experiment in sweep0
    dict['exp_id'] = expid
    all_paramdicts.append(dict)
if len(all_paramdicts) > 999: raise ValueError(f"Too many experiments in sweep. Need to add more digits to exp_id to accomodate.")
print(f"RUNNING SWEEP OF {len(all_paramdicts)} EXPERIMENTS\n", flush=True)

# Create sweep output directory and save the base file, diff dile, and list of param dicts
sweep_dir = osp.expanduser(f"{basedict['task_dir']}/sweep") 
os.makedirs(sweep_dir)
shutil.copy(osp.expanduser(args.f), sweep_dir)
shutil.copy(osp.expanduser(args.d), sweep_dir)
pickle.dump(all_paramdicts, open(f"{sweep_dir}/paramdicts.pkl", "wb"))

# # Create temporary output directory for threads to deposit configs, trained modls, and results files using temporary IDs 
# procs_outdir = f"{out_pointer}/procs_temp" 
# if osp.exists(procs_outdir): 
#     raise ValueError(f"Temporary output directory {procs_outdir} already exists. Should delete before adding more experiments to sweep.")
# print(f"Creating temporary output directory {procs_outdir} for each thread to store tree and meta files.")
# os.makedirs(procs_outdir)   

# Set up multithreading processes            
n_threads = 10  
paramdict_sets = np.array_split(all_paramdicts, n_threads)
print(f"Splitting the {len(all_paramdicts)} param sets into {len(paramdict_sets)} sets of ~{len(paramdict_sets[0])} files each, to be processed by {n_threads} threads.")

# Run each experiment by passing the sets of parmater sets to the threads
pool = mp.Pool(processes=n_threads) 
results = []
for i in range(len(paramdict_sets)):
    print(f"Starting thread {i} with {len(paramdict_sets[i])} experiments.")
    paramdicts = paramdict_sets[i]
    args = (paramdicts, sweep_dir, i)
    result = pool.apply_async(run_utils.run_sweep_proc, args=args)#), callback=response)
    results.append(result.get())
pool.close()
pool.join()
if len(results) != 0:
    for i in range(len(results)):
        print(f"proc {i}: {results[i]}")
    raise ValueError('Some threads failed. Check above for error messages.')

# # Move the configs, trained models, and results files from the temporary threads directory into the sweep folder with permanent unique IDs
# print(f"\nDONE ALL THREADS. Moving outputs from temporary dir to sweep dir.\n")
# run_utils.move_thread_outputs(procs_outdir, out_pointer)
print(f"DONE")
