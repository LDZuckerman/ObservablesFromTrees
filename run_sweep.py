'''
NOTE: For any keys swept over (e.g. in diff file), any values set for those keys in the base file are overwritten in each run.
      However, the base file will not be modified.
      This is to allow for an existing exp file to be used as the base file. 
'''

import os, json, shutil, argparse
import os.path as osp
import pandas as pd
import numpy as np
import dev.train_script as train_script
import dev.run_utils as run_utils

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-d", "--d", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False, default='1')
parser.add_argument("-redo", "--redo", type=str, required=False, default='False')
parser.add_argument("-resume", "--resume", type=str, required=False, default='False')
args = parser.parse_args()

# Load base dictionary 
basefile = osp.expanduser(args.f) # osp.join(osp.abspath(''), args.f)
basedict = json.load(open(basefile))

# Get param sets list from diff dictionary
difffile = osp.expanduser(args.d)
diffdict = json.load(open(difffile))
diffkeys = list(diffdict.keys())
sets = run_utils.perms(diffdict) 
print(f"RUNNING SWEEP OF {len(sets)} EXPERIMENTS\n")
print(f"Exploring the following paramsters: {diffkeys}")
sets = pd.DataFrame(columns = list(diffdict.keys()), data = sets)

# Set output directory: one for entire sweep, not for each exp
try: 
    swp_id = basedict['swp_id'] # if sweep id is specified in basefile, use that
    if 'sweep' not in swp_id: raise ValueError(f"WARNING: sweep id should contain 'sweep'.")
except:
    swp_id =f"sweep{basedict['exp_id'][3:]}" # if using previous exp file as basefile (e.g. if using exp9/expfile.json, swp_id = sweep9)
out_pointer = osp.expanduser(f"{basedict['task_dir']}{swp_id}") 
if eval(args.redo):
    print(f"Redo flag is True, deleting output directory {out_pointer}/")
    shutil.rmtree(out_pointer)
if not osp.exists(out_pointer):
    print(f"Makeing output directory {out_pointer}/")
    os.makedirs(out_pointer)
elif osp.exists(out_pointer) and not eval(args.resume) and not eval(args.redo):
    raise ValueError(f"   CUATION: Sweep output directory {out_pointer}/ already has some models in it, and both redo and resume flags are False.")

# Copy base file and diff file to output directory
shutil.copy(basefile, osp.join(out_pointer, 'basefile.json'))
shutil.copy(difffile, osp.join(out_pointer, 'difffile.json'))

# Create file to save done experiments
donefile = osp.join(out_pointer, 'done_experiments.json')
print(f"Creating {donefile} to save done experiments")
json.dump([], open(donefile, 'w'))

# If resume, find the last experiment done and start from the next one
if eval(args.resume):
    sets = run_utils.find_next_exp(out_pointer, donefile, sets)

# Loop over experiments prescribed by diff file
count = 0
for i in range(len(sets)):

    # Create new dict and set unique exp id
    dict = basedict.copy()
    expnum = run_utils.get_swp_num(out_pointer) # This should also allow for resuming
    dict['exp_id'] = f"{swp_id}_{expnum}"
    print(f"\nSETTING UP EXPERIMENT {dict['exp_id']}\n') # print(f'\nParameters are the following -> \n{dict}")

    # Modify dict to use params in set, and to ensure low output
    set = sets.iloc[i]
    print(f"Modified keys for this run: {set.to_dict()}")
    for groupkey in ['data_params', 'learn_params', 'hyper_params', 'run_params']:
        for key in basedict[groupkey].keys():
            if key in diffkeys:
                dict[groupkey][key] = set[key]
    dict['run_params']['performance_plot'] = None
    dict['run_params']['save_plot'] = False 
    dict['run_params']['log'] = False 
    dict['run_params']['n_trials'] = 1

    # Train (train and val)
    train_script.run(dict['run_params'], dict['data_params'], dict['learn_params'], dict['hyper_params'], out_pointer, dict['exp_id'], args.gpu, low_output=True)

    # Save experiment to done file
    dicts = json.load(open(donefile, 'r'))
    dicts.append(dict)
    json.dump(dicts, open(donefile, 'w')) # open file twice to ensure overwrite
    count += 1
    print(f'DONE with {expid} [{count}/{len(sets)}]\n')