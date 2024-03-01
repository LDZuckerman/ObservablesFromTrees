import os, json, shutil, argparse
import os.path as osp
from dev.train_script import run

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-DS_name", "--DS_name", type=str, required=True)
parser.add_argument("-outdir", "--outdir", type=str, required=False, default='~/ceph/Data/')
parser.add_argument("-ctrees_path", "--ctrees_path", type=str, required=False, default='/mnt/sdceph/users/sgenel/IllustrisTNG/L75n1820TNG_DM/postprocessing/trees/consistent-trees/')
parser.add_argument("-groupcat_path", "--groupcat_path", type=str, required=False, default='/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/L75n1820TNG/output/')
parser.add_argument("-featnames", "--featnames", type=str, required=False, default=['#scale(0)', 'desc_scale(2)', 'num_prog(4)', 'phantom(8)', 'Mvir(10)'])
args = parser.parse_args()

# Set paths 
ctrees_path = args.ctrees_path
obscat_path = args.groupcat_path
featnames = args.featnames
outpath = f'{args.outdir}/{args.DS_name}_'
metafile = f'{outpath}meta.json'
photfile = f'{outpath}allphot.pkl'
treefile = f'{outpath}alltrees.pkl'

# Prepare and save photometric data
if not os.path.exists(photfile):  
    funclib.prep_phot(obscat_path, metafile=metafile, save_path=photfile)
with open(photfile, 'rb') as f:
    allphot = pickle.load(f)
print(allphot);a=b

## Prepare and save tree data
# if not os.path.exists(treefile):
#     funclib.prep_trees(ctrees_path, featnames, phot_ids=list(allphot.keys()), metafile=metafile, save_path=treefile, df=pd0); a =b  
# with open(treefile, 'rb') as f:
#     alltrees = pickle.load(f)
# CHECK TREES USING NETWORK X PACKAGE AND/OR analysis/visualise_tree.ipynb HIERARCHY_POS 

# Prepare and save graphs
funclib.make_graphs(alltrees, allphot, featnames, metafile, save_path=f'~/ceph/Data/{DS_name}.pkl') # transformer = skp.QuantileTransformer(n_quantiles=10, random_state=0)
