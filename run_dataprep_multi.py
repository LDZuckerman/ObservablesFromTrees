import os
import os.path as osp
import argparse
import pickle, json
import numpy as np
from datetime import date
from dev import data_utils
import multiprocessing as mp

'''
Same as run_dataprep.py but with multiprocessing
'''

# Read in arguements 
parser = argparse.ArgumentParser()
parser.add_argument("-DS_name", "--DS_name", type=str, required=True)
parser.add_argument("-out_basepath", "--out_basepath", type=str, required=False, default='~/ceph/Data/')
parser.add_argument("-tng_vol", "--tng_vol", type=str, required=True)
parser.add_argument("-downsize_method", "--downsize_method", type=str, required=False, default=3)
parser.add_argument("-tinytreetest", "--tinytreetest", type=bool, required=False, default=False)
parser.add_argument("-sizelim", "--sizelim", required=False, default='None')
parser.add_argument("-reslim", "--reslim", required=False, default=100)
parser.add_argument("-transform_name", "--transform_name", type=str, required=False, default='QuantileTransformer')
args = parser.parse_args()
sizelim = np.inf if args.sizelim == 'None' else int(args.sizelim)

# Set paths 
outdir = osp.expanduser(f'{args.out_basepath}/vol{args.tng_vol}/')
if not osp.exists(outdir): 
    print(f'Creating output directory {outdir}', flush=True)
    os.makedirs(outdir)
outpath = osp.expanduser(f'{outdir}{args.DS_name}')
metafile = osp.expanduser(f'{outpath}_meta.json')
obsfile = osp.expanduser(f'{outpath}_allobs.pkl')
treefile = osp.expanduser(f'{outpath}_alltrees.pkl')
graphfile = osp.expanduser(f'{outpath}.pkl')
volnames = {'100':'L75n1820TNG', '300':'L205n2500TNG', '50':'L35n2160TNG'}
ctrees_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volnames[args.tng_vol]}_DM/postprocessing/trees/consistent-trees/'
obscat_path = f'/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/{volnames[args.tng_vol]}/output/'
crossmatchRSSF_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volnames[args.tng_vol]}_DM/postprocessing/trees/rockstar/matches/rockstar_subhalo_matching_to_FP.hdf5'
rstar_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volnames[args.tng_vol]}_DM/postprocessing/trees/rockstar/groups_099/fof_subhalo_tab_099.0.hdf5'

# Hardcoded column names
allnames = ['#scale(0)', 'id(1)', 'desc_scale(2)', 'desc_id(3)', 'num_prog(4)', 'pid(5)', 'upid(6)', 'desc_pid(7)', 'phantom(8)', 'sam_Mvir(9)', 'Mvir(10)', 'Rvir(11)', 'rs(12)', 'vrms(13)', 'mmp?(14)', 'scale_of_last_MM(15)', 'vmax(16)', 'x(17)', 'y(18)', 'z(19)', 'vx(20)', 'vy(21)', 'vz(22)', 'Jx(23)', 'Jy(24)', 'Jz(25)', 'Spin(26)', 'Breadth_first_ID(27)', 'Depth_first_ID(28)', 'Tree_root_ID(29)', 'Orig_halo_ID(30)', 'Snap_idx(31)', 'Next_coprogenitor_depthfirst_ID(32)', 'Last_progenitor_depthfirst_ID(33)', 'Last_mainleaf_depthfirst_ID(34)', 'Tidal_Force(35)', 'Tidal_ID(36)', 'Rs_Klypin', 'Mvir_all', 'M200b', 'M200c', 'M500c', 'M2500c', 'Xoff', 'Voff', 'Spin_Bullock', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]', 'A[z]', 'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', 'A[y](500c)', 'A[z](500c)', 'T/|U|', 'M_pe_Behroozi', 'M_pe_Diemer', 'Halfmass_Radius']
idnames = ['id(1)', 'desc_id(3)', 'pid(5)', 'upid(6)', 'desc_pid(7)', 'phantom(8)', 'Breadth_first_ID(27)', 'Depth_first_ID(28)', 'Tree_root_ID(29)', 'Orig_halo_ID(30)', 'Snap_idx(31)', 'Next_coprogenitor_depthfirst_ID(32)', 'Last_progenitor_depthfirst_ID(33)', 'ast_mainleaf_depthfirst_ID(34)','Tidal_ID(36)'] # all cols that are IDs
ignorenames = ['sam_Mvir(9)', 'M_pe_Behroozi'] #'Voff'] # Puebla, 2016 says dont use sam_Mvir. M_pe_Behroozi give issues in tree_3_4_1. V_off in 3_1_2
featnames = [name for name in allnames if name not in idnames + ignorenames] # all cols that are features

# Dump some dataset meta
listmeta = [{'set meta': {'Volume':args.tng_vol, 'TreeSizeLim':args.sizelim, 'Date':str(date.today())}}]
if not osp.exists(metafile):
    with open(metafile, 'w') as f:
        json.dump(listmeta, f) 
    f.close()

# Prepare and save photometric data
if not osp.exists(obsfile):  
    print(f'Preparing photometric data, will save to {obsfile}', flush=True)
    if args.testobs: data_utils.prep_mstar(obscat_path, save_path=obsfile)
    else: data_utils.prep_phot(obscat_path, crossmatchRSSF_path, rstar_path, metafile=metafile, save_path=obsfile, reslim=int(args.reslim))
else: 
    print(f'Observational data already exists in {obsfile}', flush=True)

# Prepare and save tree data 
if not osp.exists(treefile):
    print(f'Preparing tree data, will save to {treefile}', flush=True)
    print(f'\tLoading photometric data from {obsfile}', flush=True)
    with open(obsfile, 'rb') as f:
        allobs = pickle.load(f)
    data_utils.prep_trees(ctrees_path, featnames, phot_ids=list(allobs.keys()), metafile=metafile, save_path=treefile, sizelim=sizelim, tinytest=args.tinytreetest, downsize_method=args.downsize_method)
else:
    print(f'Tree data already exists in {treefile}', flush=True)

# Prepare and save graphs
if not args.testobs:
    print(f'Preparing graphs, will save to {graphfile}', flush=True)
    print(f'\tLoading obs data from {obsfile}', flush=True)
    with open(obsfile, 'rb') as f:
        allobs = pickle.load(f)
    print(f'\tLoading tree data from {treefile}', flush=True)
    with open(treefile, 'rb') as f:
        alltrees = pickle.load(f)
    data_utils.make_graphs(alltrees, allobs, featnames, metafile, save_path=graphfile) # transformer = skp.QuantileTransformer(n_quantiles=10, random_state=0)
