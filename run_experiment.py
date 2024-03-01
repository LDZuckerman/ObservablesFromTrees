import os, json, shutil, argparse
import os.path as osp
from dev.train_script import run

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)
args = parser.parse_args()

# Load construction dictionary from json file
exp_file = osp.join(osp.abspath(''), args.f)
construct_dict = json.load(open(exp_file))
#construct_dict['experiment_name']=args.f[:-5]

# Run experiment (train and val for each of n_trials)
print(f'\nSETTING UP EXPERIMENT FROM {exp_file}')
print(f'\nParameters are the following -> \n{construct_dict}')
run(construct_dict, args.gpu)

# # Move experiment file to "done" if desired
# if construct_dict['move']:
#     shutil.move(exp_file, osp.join(exp0_folder+"/done", experiment))

#print(f"Experiment {experiment} done \t {experiment}: {i + 1} / {len(exp_list)}")
