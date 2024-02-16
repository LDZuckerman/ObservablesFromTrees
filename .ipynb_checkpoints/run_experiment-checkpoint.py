import os, json, shutil, argparse
import os.path as osp
from dev.utils import list_experiments

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)
args = parser.parse_args()

# Set training script based on gpu use
if args.gpu=='1':
    from dev.train_script_gpu import train_model
else:
    from dev.train_script_cpu import train_model

# Get list of "experiment" json files from experiment todo folder
exp_folder, exp_list = list_experiments(str(args.f))
print(f"Starting process with {len(exp_list)} experiments ({exp_list}) from '{str(args.f)}/todo/'" )

# Loop over experiments
for i, experiment in enumerate(exp_list):
    print(f'Running experiment {i} from {experiment})
          
    # Load construction dictionary from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    construct_dict['experiment_name']=experiment[:-5]
    
    # Do training 
    train_model(construct_dict)

    # Move experiment file to "done" if desired
    if construct_dict['move']:
        shutil.move(osp.join(exp_folder, experiment), osp.join(exp0_folder+"/done", experiment))
    print(f"Experiment {experiment} done \t {experiment}: {i + 1} / {len(exp_list)}")
