'''
Save exp.json file in parent directory of ObservablesFromTrees
In parent directory, run 'python ObservablesFromTrees/run_experiment.py -f exp.json -gpu 1'
'''

print(f"Running {__file__}")

import os, json, shutil, argparse
import os.path as osp
import dev.train_script as train_script

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)
parser.add_argument("-redo", "--redo", type=str, required=False, default='False')
args = parser.parse_args()

# Load construction dictionary from exp json file
exp_file = osp.expanduser(args.f) # osp.join(osp.abspath(''), args.f)
dict = json.load(open(exp_file))
print(f"Experiment parameters: \n{dict}\n")
print(f'\nSETTING UP EXPERIMENT\n') # print(f'\nParameters are the following -> \n{d}')

# Set up output and copy experiment json file to output dir 
out_pointer = osp.expanduser(f"{dict['task_dir']}{dict['exp_id']}") # Dir for saving all training outputs (if n_trial > 1, will create subfolders. Will also have subfolder for TF logging)
if not osp.exists(out_pointer):
    print(f"Makeing output directory {out_pointer}/")
    os.makedirs(out_pointer)
elif osp.exists(out_pointer) and osp.exists(osp.join(out_pointer, 'result_dict.pkl')) and not eval(args.redo):
    raise ValueError(f"   CUATION: Output directory {out_pointer}/ already seems to already have a trained model in it, and redo flag is False.")
elif osp.exists(out_pointer) and osp.exists(osp.join(out_pointer, 'result_dict.pkl')) and eval(args.redo):
    print(f"   Redo flag is True, deleting result_dict.pkl and mode_best.pt from exp folder")
    os.remove(osp.join(out_pointer, 'result_dict.pkl'))
    os.remove(osp.join(out_pointer, 'model_best.pt'))
try: 
    shutil.copy(exp_file, out_pointer) 
    print(f"Copying experiment file into {out_pointer}/ for future reference")
except shutil.SameFileError: 
    print(f"Using experiment file in {out_pointer}/")


# Train (train and val for each of n_trials)
train_script.run(dict['run_params'], dict['data_params'], dict['learn_params'], dict['hyper_params'], out_pointer, dict['exp_id'], args.gpu)
print('DONE')


