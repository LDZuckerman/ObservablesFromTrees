from calendar import firstweekday
from curses.ascii import RS
import pickle, time, os
from flask import g
import numpy as np
import os.path as osp
from datetime import date, datetime
from sympy import half_gcdex

import torch
from torch import half
os.environ['TORCH'] = torch.__version__
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset as Dataset_notgeom

import illustris_python as il
import pandas as pd
import h5py
from IPython.display import display
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from itertools import count
import io
from sklearn.model_selection import train_test_split

# Make dictionary of (cm_ID, tree DF) for all trees
def prep_trees(ctrees_path, featnames, phot_ids, metafile, save_path, zcut=('before', np.inf), Mlim=10, sizelim=np.inf, tinytest=False, downsize_method=3): 
          
    # Prepare to load tree data 
    tree_path = osp.expanduser(ctrees_path)
    #loadcols = [allnames.index(name) for name in allnames if name not in ignorenames] # col idxs to load (remove ignorenames)
    print(f'\tTree features to include: {featnames}', flush=True)
    downsize_func = downsize_tree3 if downsize_method == 3 else downsize_tree1 if downsize_method == 1 else None
    print(f'\tWill use downsize method {downsize_method}', flush=True)

    # Initialize meta dict and tree dicts
    all_trees = {}
    meta = {'featnames': featnames,
            'zcut': str(zcut), 'Mlim': str(Mlim), 'sizelim': str(sizelim),
            'tot trees': 0, 'tot trees saved': 0,
            'trees mass cut': 0, 'trees sat cut': 0, 'trees size cut': 0, 'trees no mergers cut': 0,
            'downsize method': downsize_method,
            'start stamp': str(datetime.now())}
    
    # Add tree dataframes to all_trees dict
    t0 = time.time()
    fcount = 0
    treefiles = [f for f in os.listdir(tree_path) if f.endswith('.dat') and f.startswith('tree_')]
    for treefile in treefiles:
                
        # Load tree sets (each file has many trees, with a tree being a header line followed by rows of nodes)
        print(f'\tLoading trees from {treefile}', flush=True)
        t0_file = time.time()
        pd1 = pd.read_table(f'{tree_path}/{treefile}', skiprows=0, delimiter='\s+', dtype=str) # read ALL COLS in (as strings) 
        raw = pd1.drop(axis=0, index=np.arange(48)) # remove rows 0 to 48, which are one large set of comments
        del pd1
        
        # Split into df of just trees by detecting the tree header lines (they have NaN in all but the first two cols)
        halos = raw[~raw.isna()['desc_id(3)']] # rows (nodes)
        del raw
        
        # For all halos in this file, change dtypes, make z cut, scale columns
        #print('\t\tChanging dtypes, cutting in z, scaling columns', flush=True)
        halos = change_dtypes(halos, featnames) # only change for features (keep ids as strings)
        halos = make_zcut(halos, zcut)
        halos, meta = scale(halos, featnames, meta) # REMOVED num_prog FROM COLS GETTING LOG SCALED 

        # For each tree, construct dataframe, downsize, save to tree dict with corresponding crossmatch ID as key
        indices = [i for i, x in enumerate(halos['desc_id(3)']) if x == '-1'] # (3/5) THIS WAS NOT '-1' BEFORE! Indices where desc_id is -1 (first line of each tree)
        split_halos = np.split(np.array(halos), np.array(indices[1:])) # list of 2d tree arrays #print(f'\t\tSplitting into {len(split_halos)} individual trees', flush=True)
        meta['tot trees'] += len(split_halos)
        tcount = 0
        for i in range(len(split_halos)): 
            haloset = split_halos[i]
            tree = pd.DataFrame(data = haloset, columns = halos.columns)
            halo_z0 = tree.iloc[0]
            rstar_id = str(int(halo_z0['Orig_halo_ID(30)']))
            # Ignore trees that trace history of non-central, low-mass, or low_resolution halos. 
            lowmass = halo_z0['Mvir(10)'] <= Mlim # if final halo too low mass
            satelite = halo_z0['pid(5)'] != '-1' # if not central
            SFcut = rstar_id not in phot_ids # if cut based on SF criteria (e.g. not central according to SF, or low resolution)
            toolong = len(haloset) > sizelim # Christian cut out those over 20000, but Rachel says not to
            nomergers = len(haloset[haloset[:,4]>1]) == 0
            if (toolong or lowmass or satelite or SFcut or nomergers): 
                #reason = f"too many halos ({len(haloset)})" if toolong else f"not central (pid = {halo_z0['pid(5)']})" if satelite else f"not massive enough (Mvir = {halo_z0['Mvir(10)']})" if lowmass else "not central according to SF" if SFcut else else "no mergers" if nomergers else "SHOULD NOT HAVE BEEN CUT"
                # print(f"\t\t\tIgnoreing haloset {i}") #becasue {reason}", flush=True)
                if lowmass: meta['trees mass cut'] += 1
                if satelite: meta['trees sat cut'] += 1
                if toolong: meta['trees size cut'] += 1
                if nomergers: meta['trees no mergers cut'] += 1
                continue
            # print(f'\t\t\tProcessing haloset {i} into tree', flush=True)
            tcount += 1
            # Remove "unimportant" subhalos and add newdesc_id with updated descendents
            dtree = downsize_func(tree, num=i)
            dtree = dtree.rename(columns={'desc_id(3)': 'olddesc_id'}) # rename desc_id to olddesc_id
            # Add (rstar_id , tree) to dicts
            all_trees[rstar_id] = dtree
        
        fcount += 1
        meta['tot trees saved'] += tcount

        print(f"\tDone with {treefile} ({fcount} of {len(treefiles)}). Took {np.round(time.time()-t0_file, 4)}s. Processed {tcount} of {len(split_halos)} trees. {meta['trees mass cut']} mass cut, {meta['trees sat cut']} sats, {meta['trees size cut']} too big, {meta['trees no mergers cut']} no mergers.", flush=True)

        if tinytest: break # for small set to test on, just use trees in one file

    meta['total time'] = np.round(time.time() - t0, 4)
    print(f"\tDone with all files. Took {np.round(time.time() - t0, 4)} s to save {meta['tot trees saved']} total trees")
    print(f'\tSaving trees in {save_path} and adding meta to {metafile}', flush=True)  
    with open(save_path, 'wb') as f:
        pickle.dump(all_trees, f)
    # with open(osp.expanduser('~/ceph/Data/temptreemetafile.pkl'), 'w') as f:
    #     json.dump({'tree meta':meta}, f)
    with open(metafile, 'r+') as f: # with w+ I cant read it but if I write to it with r+ I cant read the new file
        listmeta = json.load(f)
    if list(listmeta[-1].keys())[0] == 'tree meta': # If tree meta already exists (this is a re-do), remove old tree meta
        listmeta = listmeta[:-1]
    listmeta.append({'tree meta':meta})  
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)
    f.close()
    
    return 

# Make dictionary of (subhalo, phot) for all central subhalos
def prep_phot(obscat_path, crossmatchRSSF_path, rstar_path, metafile, save_path, reslim=100):

    meta = {'label_names': ["U", "B", "V", "K", "g", "r", "i", "z"], 'reslim': reslim, # Note: had a typo swapping V and B, but I've fixed that
            'tot galaxies': 0, 'tot galaxies saved': 0, 
            'galaxies res cut': 0, 'galaxies sat cut': 0, 
            'start stamp': str(datetime.now())}

    # Load files for crossmatching to rstar IDs
    cmRSSF_list = h5py.File(crossmatchRSSF_path, 'r')['Snapshot_99']['SubhaloIndexDarkRockstar_SubLink'][:]
    rstar_subhalo_ids = np.array(h5py.File(rstar_path, 'r')['Subhalo']['Subhalo_ID'])

    # Load subhalo photometry and resolution
    subhalos = il.groupcat.loadSubhalos(basePath = obscat_path, snapNum=99)
    sh_phot = subhalos['SubhaloStellarPhotometrics']
    sh_res = subhalos['SubhaloLenType'][:,4] # 4 for star particles 
    meta['tot galaxies'] = len(sh_phot)

    # Get central subhalo indices
    groups = il.groupcat.loadHalos(basePath = obscat_path, snapNum=99)
    ctl_idxs = groups['GroupFirstSub'] # for each group, the idx w/in "subhalos" that is its central subhalo (-1 if group as no subhalos)
    ctl_idxs = ctl_idxs[ctl_idxs != -1] # remove groups with no subhalos
    meta['galaxies sat cut'] = len(sh_phot) - len(ctl_idxs)
    print(f'\tTotal of {len(sh_phot)} subhalos, with {len(ctl_idxs)} being central', flush=True)  

    # Add photometry for all central subhalos to dict
    all_phot = {}
    for idx in ctl_idxs:
        # Remove if not central or resolution too low
        if (sh_res[idx] < reslim): 
            meta['galaxies res cut'] += 1
            continue
        # Get rstar_id for matching to tree
        subfind_read_idx =  idx
        rstar_read_idx = cmRSSF_list[subfind_read_idx]
        rstar_id = int(rstar_subhalo_ids[rstar_read_idx])
        # Add phot to dict under rstar_id
        all_phot[str(rstar_id)] = sh_phot[idx] 
        meta['tot galaxies saved'] += 1

    # Save to pickle
    print(f'\tSaving phot in {save_path} and meta in {metafile}', flush=True)     
    with open(save_path, 'wb') as f:
        pickle.dump(all_phot, f)
    #listmeta = [{'obs meta':meta}]
    # with open(metafile, 'w') as f:
    #     json.dump(listmeta, f) 
    with open(metafile, 'r+') as f: # with w+ I cant read it but if I write to it with r+ I cant read the new file
        listmeta = json.load(f) 
    if list(listmeta[-1].keys())[0] == 'obs meta': # If graph meta already exists (this is a re-do), remove old obs meta
        listmeta = listmeta[:-1]
    listmeta.append({'obs meta':meta}) 
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)
    f.close()
    
    return all_phot        

# Combine prepared trees and TNG subfind photometry into pytorch data objects
def make_graphs(alltrees, allphot, featnames, metafile, save_path): 
    '''
    Combine trees and corresponding targets into graph data objects
    NOTE: decide if best to apply fitted transformer here, or later (not going to do it in prep_trees to allow more flexibility')
    '''
    
    meta = {'num graphs':0, 'start stamp': str(datetime.now())}
    start = time.time()

    # Combine trees and corresponding targets into data object  # data_z.py 205 - 295
    print(f'\tCombining trees and targets ({len(list(alltrees.keys()))} total) into full data object', flush=True)
    dat = []
    count = 1
    for rstar_id in list(alltrees.keys()): # Loop through tree keys becasue tree dict wont have any keys not in obs dict
        tree = alltrees[str(rstar_id)]
        #print(f"\t   Tree {count} (rockstar ID {rstar_id}), {len(tree)} halos", flush=True); count += 1
        # Create x tensor
        data = np.array(tree[featnames], dtype=float)
        X = torch.tensor(data, dtype=torch.float) 
        # Create y tensor 
        targs = allphot[rstar_id]
        y = torch.tensor(targs, dtype=torch.float) 
        # Create edge_index tensor # Should just have one tuple for every prog in the tree (all haloes except last), right? Like, should be n_halos - 1 progs, right?
        progs = []
        descs = []
        for i in range(len(tree)):
            halo = tree.iloc[i]
            desc_id = halo['newdesc_id'] 
            if (desc_id != '-1'): # this halo has a desc (not the final halo)
                # try: print(f"\t      halo {i}, desc_id {desc_id}, desc_pos {np.where(tree['id(1)']==desc_id)}")
                # except: print(f"\t      halo {i}, desc_id {desc_id}")
                desc_pos = np.where(tree['id(1)']==desc_id)[0][0] # only reason that this halos descid would NOT be in halo ids would be if somehow its desc got cut and its descid wasnt updated
                progs.append(i)
                descs.append(desc_pos) 
        # print(f'\t\t\tshape X {X.shape} (should be [{len(tree)}, {len(featnames)}])', flush=True)
        # print(f'\t\t\tshape y {y.shape} (should be [1, 8])', flush=True)
        # print(f'\t\t\tlen(progs) {len(progs)} (should be {len(tree) - 1})', flush=True)
        edge_index = torch.tensor([progs,descs], dtype=torch.long) 
        # Create edge_attr tensor
        edges = np.full(len(progs),np.NaN)
        edge_attr = torch.tensor(edges, dtype=torch.float)
        # Combine into graph and add graph dat list
        graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dat.append(graph)  
        meta['num graphs'] += 1 
        #print(f'\t\tDone with tree. Time elapsed {time.time()-start} s', flush=True)
                
    # Save pickled dataset
    print(f'Saving dataset to {save_path} and adding meta to {metafile}', flush=True) 
    with open(osp.expanduser(save_path), 'wb') as handle:
        pickle.dump(dat, handle)
    with open(metafile, 'r+') as f: # with w+ I cant read it but if I write to it with r+ I cant read the new file
        listmeta = json.load(f) 
    if list(listmeta[-1].keys())[0] == 'graph meta': # If graph meta already exists (this is a re-do), remove old graph meta
        listmeta = listmeta[:-1]
    listmeta.append({'graph meta':meta}) 
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)

    f.close()

    return dat

 # Collect other targets for the same galaxies as in the photometry
def collect_othertargs(obscat_path, alltrees, volname, save_path, reslim=100):
    '''
    The graphs in allgraphs are a SUBSET of the galaxies in all_obs, so this will be tricky
    Recreate prep_phot but return other SG targs instead (but still with rstar idx as key)
    Then receate make_graphs, but instead of making the graphs, just stack the targs for the gals with the rstar idxs that are in alltrees
    Then acces them at testidxs in testidx file 
    NOTE: This would be similar to what I'd need to do to construct a DS on other obs but using the same trees (tree generation is the slow part)
      Difference is that right now I am not going to save full graphs, just save the targs in the correct order
      Also, instead of saving dict of {rstar_id:, targs}, I want to be able to easily access multiple targs, so Im gonna store a df with cols being rstar id and then targ names
      Note that for the future need to ensure run_dataprep.py and stuff are felxible about using the same graph sets but different targs (maybe they are already)
    '''

    # Load files for crossmatching to rstar IDs
    crossmatchRSSF_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volname}_DM/postprocessing/trees/rockstar/matches/rockstar_subhalo_matching_to_FP.hdf5'
    cmRSSF_list = h5py.File(crossmatchRSSF_path, 'r')['Snapshot_99']['SubhaloIndexDarkRockstar_SubLink'][:]
    rstar_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volname}_DM/postprocessing/trees/rockstar/groups_099/fof_subhalo_tab_099.0.hdf5'
    rstar_subhalo_ids = np.array(h5py.File(rstar_path, 'r')['Subhalo']['Subhalo_ID'])

    # Targets to load
    to_store = {'SubhaloBHMass': r'Black Hole Mass [$10^{10}$M$_{\odot}/h$)]', # why would this be different than SubhaloMassType[5]?
                'SubhaloBHMdot': r'Black Hole Accretion Rate [$10^{10}$M$_{\odot}/h$/yr]', 
                'SubhaloGasMetallicity': 'Gas Metallicity', 
                'SubhaloHalfmassRad': 'Halfmass Radius [ckpc/h]',
                'SubhaloMass': r'Total Mass [$10^{10}$M$_{\odot}/h$]', 
                'SubhaloGasMass': r'Gass Mass [$10^{10}$M$_{\odot}/h$]',
                'SubhaloStelMass': r'Stellar Mass [$10^{10}$M$_{\odot}/h$]',
                'SubhaloSFR': r'Star Formation Rate [M$_{\odot}$/yr]', 
                'SubhaloStarMetallicity': r'Stellar Metallicity [M$_z$/M$_{tot}$]', 
                'SubhaloVelDisp': r'Velocity Dispersion [km/s]', 
                'SubhaloVmax': r'V_${max}$ [km/s]',
                'SubhaloRes': 'Resolution',
                }

    # Load a variety of subhalo targets, and resolution
    subhalos = il.groupcat.loadSubhalos(basePath = obscat_path, snapNum=99)
    sh_res = subhalos['SubhaloLenType'][:,4] # 4 for star particles 

    # Get central subhalo indices
    groups = il.groupcat.loadHalos(basePath = obscat_path, snapNum=99)
    ctl_idxs = groups['GroupFirstSub'] # for each group, the idx w/in "subhalos" that is its central subhalo (-1 if group as no subhalos)
    ctl_idxs = ctl_idxs[ctl_idxs != -1] # remove groups with no subhalos
    galaxies_sat_cut = len(sh_res) - len(ctl_idxs) # for checking that the same cuts are being applied as in prep_phot
    print(f'\tTotal of {len(sh_res)} subhalos, with {len(ctl_idxs)} being central', flush=True)  

    # Add photometry for all central subhalos to dict
    all_targs =  pd.DataFrame(columns=['rstar_id']+list(to_store.keys()))
    galaxies_res_cut = 0 # for checking that the same cuts are being applied as in prep_phot
    galaxies_saved = 0 # for checking that the same cuts are being applied as in prep_phot
    for idx in ctl_idxs:
        # Remove if not central or resolution too low, otherwise start list of values
        if (sh_res[idx] < reslim): 
            galaxies_res_cut += 1 
            continue
        # Get rstar_id for matching to tree
        subfind_read_idx = idx
        rstar_read_idx = cmRSSF_list[subfind_read_idx]
        rstar_id = int(rstar_subhalo_ids[rstar_read_idx])
        # Add targ values to list
        info = []
        info.append(str(int(rstar_id)))
        for key in to_store.keys():
            if key == 'SubhaloRes': info.append(sh_res[idx])
            elif key == 'SubhaloGasMass': info.append(subhalos['SubhaloMassType'][idx,0]) # WAS [IDX][O] BUT I DONT THINK THAT SHOULD BE THE PROBLEM
            elif key == 'SubhaloStelMass': info.append(subhalos['SubhaloMassType'][idx,4])
            else: info.append(subhalos[key][idx])
        # Add list to DF
        all_targs.loc[len(all_targs)] = info

    # Collect just the galaxies that are in the stored trees
    all_targs_out = pd.DataFrame(columns = all_targs.columns)
    for rstar_id in list(alltrees.keys()):
       all_targs_out.loc[len(all_targs_out)] = all_targs[all_targs['rstar_id'] == rstar_id].iloc[0]

    # all_targs = all_targs[all_targs['rstar_id'].isin(list(alltrees.keys()))]
    all_targs_out = all_targs_out.drop(columns=['rstar_id'])

    # Save target data and target descriptions
    print(f'\tSaving collected targets in {save_path}', flush=True)     
    with open(save_path, 'wb') as f:
        pickle.dump(all_targs_out, f)
    f.close()
    with open('../Data/subfind_othertargdescriptions.json', 'w') as f:
        json.dump(to_store, f) 

    return all_targs, galaxies_res_cut, galaxies_saved, galaxies_sat_cut

def collect_MHmetrics(testdata, used_feats, all_featnames, ft_test, save_path):
    # Should I also add metrics about SFH? That would require something similar to collect_othertargs
    # I should probabaly also be doing t_50, etc, but I think converting from z to t is nontrivial 

    all_metrics = pd.DataFrame(columns=['n_halos', 'z_25', 'z_50', 'z_75'])

    scale_col = used_feats.index('#scale(0)')
    mass_col = used_feats.index('Mvir(10)')
    for graph in testdata:
        x = graph.x.detach().numpy()
        qt = ft_test.named_transformers_['num']
        x_exp = np.zeros((len(x), len(all_featnames)))
        x_exp[:,all_featnames.index('#scale(0)')] = x[:, scale_col]
        x_exp[:,all_featnames.index('Mvir(10)')] = x[:, mass_col]
        x_exp_t = qt.inverse_transform(x_exp) # transform back to original
        x[:,scale_col] = x_exp_t[:,all_featnames.index('#scale(0)')]
        x[:,mass_col] = x_exp_t[:,all_featnames.index('Mvir(10)')]
        scales = np.unique(x[:,scale_col])
        mass_cdf = [0]
        for scale in scales:
            halos_z = x[x[:,scale_col] == scale]
            added = np.sum(halos_z[:,mass_col])
            mass_cdf.append(mass_cdf[-1]+added)
        mass_cdf = mass_cdf[1:]/np.max(mass_cdf)
        z = 1/scales - 1
        z_25 = z[np.where(mass_cdf > 0.25)[0][0]]
        z_50 = z[np.where(mass_cdf > 0.5)[0][0]]
        z_75 = z[np.where(mass_cdf > 0.75)[0][0]]
        info = [len(x), z_25, z_50, z_75]
        all_metrics.loc[len(all_metrics)] = info

    pickle.dump(all_metrics, open(save_path, 'wb'))

    return all_metrics


# Change dtypes of first 25 cols (which probably by error are all strings)
def change_dtypes(halos, featnames):
    row = halos.iloc[0]
    castto = []
    for featname in featnames:
        # dtype = "int64" if not isinstance(row[featname], str) else "float64" if '.' in row[featname] else "int64"
        # dtype = type(row[featname]) if not isinstance(row[featname], str) else "float64" if '.' in row[featname] else "int64"
        featvals = halos[featname]
        containsfloatstr = any('.' in val for val in featvals)
        containsnonstr = any(not isinstance(val, str) for val in featvals)
        # if containsnonstr:
        #     t = [type(val) for val in featvals if not isinstance(val, str)][0] # Lets see if i need this (like, are there any times when a row just has one or two non strings?)
        dtype = type(featvals[0]) if containsnonstr else "float64" if containsfloatstr else "int64"
        castto.append([featname, dtype])
    halos = halos.astype(dict(castto))

    return halos 

# Make cut in z
def make_zcut(halos, zcut):
    
    cut = zcut[0] # 'before' or 'after'
    z_lim = zcut[1] 
    if cut == 'before':
        if z_lim != np.inf: print('WARNING: havent fully tested zcutting.', flush=True)
        # Find halos within redshift limit
        keep = (1/halos['#scale(0)']-1) <= z_lim
        # change num prog for halos whose proginators were just cut
        cut_descids = list(halos[~keep]['desc_id(3)'])  # desc ids of halos that were cut
        for descid in cut_descids:
            curr_num = halos.loc[halos['id(1)'] == descid, 'num_prog(4)'].iloc[0]
            halos.loc[halos['id(1)'] == descid, 'num_prog(4)'] = curr_num - 1
            print(halos.loc[halos['id(1)'] == descid, 'num_prog(4)'].iloc[0], flush=True)
        halos = halos[keep]
    if cut == 'after':
        raise ValueError('Not yet set up to try "after" option -  will need to figure out if its even possible (do I have phot at z<0?)')
    
    return halos

# Scale appropriate cols
def scale(halos, featnames, meta):
    
    # Define scaling funcs
    def logt(x):
        return np.log10(x+1)
    def maxscale(x):
        return x/max(x)

    # Scale some features with logt # WHY WAS CHRISITIAN LOG SCALING NUM_PROG??
    lognames = ['Mvir(10)', 'Mvir_all', 'M200b', 'M200c', 'M2500c', 'M_pe_Behroozi', 'M_pe_Diemer'] # [10, 38, 39, 40, 4, 42, 57, 58] # cols to take logt of (take log10 of others later??)
    # lognames = ['num_prog(4)', 'Mvir(10)', 'Mvir_all', 'M200b', 'M200c', 'M2500c', 'M_pe_Behroozi', 'M_pe_Diemer'] # [10, 38, 39, 40, 4, 42, 57, 58] # cols to take logt of (take log10 of others later??)
    tolog = [name for name in featnames if name in lognames]
    for name in tolog:
        halos[name] = logt(halos[name])

    # Scale some features with minmax 
    minmaxnames = ['Jx(23)', 'Jy(24)', 'Jz(25)'] # [23, 24, 25] # cols to maxscale ("simple min/max for 1e13 scaling down"???)
    tominmax = [name for name in featnames if name in minmaxnames]
    for name in tominmax:
        halos[name] = maxscale(halos[name])
    
    meta['logcols'] = lognames
    meta['minmaxcols'] = minmaxnames
    
    return halos, meta 

def downsize_tree1(tree): # TAKE ONLY IMPORTANT HALOS (HALOS WITH NO PROGS OR MUTILE progs, AND HALOS WITH MERGED HALOS AS THIER DESCENDENT) THIS IS s=tree[np.logical_or(tree[:,3] == -1,tree[:,4]!=1)] # 
    '''
    Halo is important if:
        - Its an root halo (n_progs = 0)
        - Its a merger baby (n_progs > 1)
        - Its one of the progs of a merger baby (merges with another halo so they share the same desc)
        - Its the z=0 halo (desc_id = -1)
    '''
    print(f"\t\t\t   Downsizing ({len(tree)} -->", end='', flush=True)
    n_rem = 0
    tree.insert(len(list(tree.columns)), 'newdesc_id', tree['desc_id(3)']) # add newdesc_id column, which will stay desc_id unless updated below
    halo_ids = list(tree['id(1)']) # iterate though ids to ensure not making mistakes (not that these are strings)
    # Loop through halo_ids 
    for hid in halo_ids:
        halo = tree[tree['id(1)']==hid]
        hdid = halo['newdesc_id'].iloc[0] # need .iloc[0] because without it it returns index too. # if np.isnan(float(hdid)): print(f"\thdid is nan for halo {hid}", flush=True); a=b
        prog_ids = list(tree[tree['newdesc_id'] == hid]['id(1)']) # ids of the halos that created this halo CHECK LEN SAME AS 'num_prog(4)'    # np.where(desc_ids == halo['id(1)'])[0] 
        if hdid != '-1':
            n_mergpartners = len(tree[tree['newdesc_id'] == hdid]) - 1 # the number of halos that merged to create this halo's descendent, excluding this halo 
        else: n_mergpartners = 0 # if this halo is the final halo (desc_id = -1), then it has no mergpartners (but we will keep it anyway due to desc_id criteria)
        important = len(prog_ids) == 0 or len(prog_ids) > 1 or n_mergpartners > 0 or hdid == '-1'
        if not important: 
            # print(f"\tHalo {hid}: {len(prog_ids)} progs, {n_mergpartners} mergpartners, desc_id {hdid}, not important. Changing desc_id to {hdid} for halos {list(prog_ids)})", flush=True)
            progmask = tree['id(1)'].isin(prog_ids) # true where halo is parent of this halo
            tree.loc[progmask, 'newdesc_id'] = hdid # set desc_id of these parent halos to this halos desc_id, since we are removing this halo
            tree = tree.drop(tree[tree['id(1)'] == hid].index) # tree.loc([tree['id(1)'] != hid]) # remove this halo from the tree
            n_rem += 1
    print(f"{len(tree)})", flush=True)
    
    return tree

def downsize_tree3new(tree, num, debug=False):
    '''
    Same as below, but attempt to speed up by not carrying through all the cols
     - Just keep ids of halos, not full rows
     - Then at the end just tree at those rows
    Nah.. this doesnt seem to speed things up at all
    '''

    #print(f"\t\t\t   Downsizing ({len(tree)} -->", end='', flush=True)

    nprogcol = tree.columns.get_loc('num_prog(4)')
    descidcol = tree.columns.get_loc('desc_id(3)')
    idcol = tree.columns.get_loc('id(1)')

    hals = []

    atree= np.array(tree)[:,0:5] # DONT NEED TO CARRY THROUGH FEATURES AFTER THIS, just put them back at the end
    roots = atree[atree[:,nprogcol]==0] 
    mergers = atree[atree[:,nprogcol]>1]
    final = atree[atree[:,descidcol]=='-1']
    des = []; 
    finalid = final[0][idcol] # [0] because atree[row] will give list inside a list
        
    hrc = -1; drc = -1
    if debug: print(f"\n")
    if finalid not in mergers[:,idcol]: # IF FIRST HALO (FINAL HALO) IS NOT A MERGER (ONLY 1 DESC) IT ALSO WONT BE A ROOT SO IT WILL NEVER GET ADDED... 
        hals.append(final[0][idcol]); hrc += 1
        des.append(final[0][descidcol]); drc += 1
        if debug:
            print(f"Final halo {finalid} is not a merger, so adding it here")
            print(f"  Adding its id to hals [hals row {hrc}]") 
            print(f"  Adding its desc id {final[0][descidcol]} to des [des row {drc}]")
    if debug: print(f"Looping through roots and mergers")
    for q, mid in enumerate(mergers[:,idcol]):
        if debug: print(f"Halo {mid} is merger {q} (descid {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]})")
        hals.append(mergers[q][idcol]); hrc +=1 
        if debug: print(f"  Adding its id it to hals [hals row {hrc}]") 
        k=1
        descid = atree[:,descidcol][np.where(mid==atree[:,idcol])][0] # look at this halo's descendent 
        while descid not in mergers[:,idcol] and descid!='-1': # if this halos desc is not a merger or final, find the next halo who's desc is merger or final and assign it to be this halos be this halos new desc
            proid = atree[:,idcol][np.where(descid==atree[:,idcol])][0] # get the id of the prog of this halo's descendent (want to keep the progs of mergers - could this get moved outside while loop and just get it if this halo's desc is a merger?)   # added [0] 
            descid = atree[:,descidcol][np.where(descid==atree[:,idcol])][0] # look at this halo's descendent's descendent   # added [0] # desc id of halos desc  ##new descendant id where current descendant id = halo id
            k += 1
        if k>1 and proid != finalid: # if original desc of this merger was not a merger but its new desc is (as opposed to new desc being final (this is same as descid != -1, right?))
            des.append(proid); drc += 1
            hals.append(atree[np.where(proid==atree[:,idcol])][0][idcol]); hrc +=1
            if debug:
                print(f"  Its original desc is not a merger or final, and this new desc is merger, not final")
                print(f"     Adding id of its new desc ('proid' {proid}) to des [des row {drc}]") # Add halo's new desc id at corresponding des index
                print(f"     Adding id of its new desc (halo at 'proid') to hals [hals row {hrc}]") # Add new descendent to hals
        if descid!='-1': # if desc of [this merger]/[this mergers new desc] is not final (is a merger)
            des.append(descid); drc +=1 # Add [id of desc]/[id of new desc's desc] to des at [idx correponding to this merger]/[idx corresponding to new desc] # removed [0]
            if debug: print(f"  Adding {'new' if k>1 else 'original'} descid {descid} to des [des row {drc}]") 
        else: # descid==-1, e.g. either didnt enter while loop or entered and found final
            if k == 1:
                des.append(atree[:,descidcol][np.where(mid==atree[:,idcol])][0]); drc += 1 # ADD ORIGINAL DESC??? THAT ONLY WORKS IF DIDNT ENTER WHILE (E.G. PROG OF FINAL IS MERGER, AND WE ARE ON THAT MERGER)
                if debug: print(f"  Adding original desc id {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]} to des [des row {drc}]")
            else: 
                des.append(finalid); drc += 1 # Add id of new desc's desc (final) to des at idx corresponding to new desc
                if debug: print(f"  Adding new descid (final halo {proid} to des [des row {drc}]")
                if proid != finalid: print(f"Error: entered loop and found final, but proid {proid} != finalid {finalid}"); a=b
        
        if hrc != drc: print(f"Error: hrc {hrc} != drc {drc}"); a=b

    c = 0     
    for r in roots:
        #print(f"Halo {r[idcol]} is root {c} (descid {atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0]})")
        hals.append(r[idcol]); hrc +=1 
        #print(f"  Adding it to hals [hals row {hrc}]")
        descid=atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0] 
        k=1
        while descid not in mergers[:,idcol] and descid!='-1': # could remove descid!=-1 right? root should never go directly to final
            proid=atree[:,idcol][np.where(descid==atree[:,idcol])][0] 
            descid=atree[:,descidcol][np.where(descid==atree[:,idcol])][0]
            k+=1
        if k>1 and proid!=finalid: # if it entered the above while loop (desc of this merger is not a merger?) and progenitor is not the final halo (?)
            hals.append(atree[np.where(proid==atree[:,idcol])][0][idcol]); hrc +=1 
            des.append(proid); drc += 1
            if debug:
                print(f"  Its original desc is not a merger or final, and new proid is also not final")
                print(f"     Adding halo at new 'proid' to hals [hals row {hrc}]")
                print(f"     Adding new 'proid' {proid} to des [des row {drc}]")
        if descid!='-1':
            des.append(descid); drc +=1 
            if debug: print(f"  Adding {'new' if k>1 else 'original'} {descid} to des [des row {drc}]")
        else: # descid==-1, e.g. didn't enter loop (dont need to worry about finding final as next desc of a root)
            des.append(atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0]); drc += 1
            if debug: print(f"  Adding original desc id {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]} to des [des row {drc}]") 

        if hrc != drc: print(f"Error: hrc {hrc} != drc {drc}"); a=b
        c += 1
    
    mask = tree['id(1)'].isin(hals) 
    halos = tree.loc[mask]
    tree_out = pd.DataFrame(halos, columns=tree.columns)
    tree_out['newdesc_id'] = des

    for i in range(len(tree_out)):
        if not (tree_out.iloc[i]['newdesc_id'] in list(tree_out['id(1)']) or tree_out.iloc[i]['newdesc_id'] == '-1'): # this halos original descendent was cut without updateing this halos descendent 
            print(f"\n[haloset {num}] Error at row {i}: no halos in the tree have this halo's descdendent ({tree_out.iloc[i]['newdesc_id']})");
            print(f"\nhalo: {list(tree_out.iloc[i])}")
            print(f"\tlist(tree_out['id(1)'])[0:5]: {list(tree_out['id(1)'])[0:5]}")
            print(f"\tlist(tree_out['newdesc_id'])[0:5]: {list(tree_out['newdesc_id'])[0:5]}")
            a=b

    try: 
        np.array(tree_out, dtype=float)
    except: 
        print(f"[haloset {num}] Cant set tree_out to array")
        for i in range(len(halmix)):
            try: np.array(tree_out.iloc[i], dtype=float)
            except: 
                print(f"[haloset {num}] Error at row {i}")
                print(f"halmix[i] \n{halmix[i]} \nhalmix[i+1] \n{halmix[i+1]}")
                print(f"tree_out.iloc[i] \n{list(tree_out.iloc[i])} \ntree_out.iloc[i+1] \n{list(tree_out.iloc[i+1])}")
                a=b

    #print(f"{len(halmix)})", flush=True) 

    return tree_out

def downsize_tree3(tree, num, debug=False):
    '''
    Attempt to replicate Christian's method
    WTF why is this so much faster its like even way faster than Christians??? Cuts the same num halos as Christians (both way more than mine)
    NOTE: NOT updating desc_scale, since probabaly more important to know z of original descendent than z of new descendent (wait - why is this any different information than scale anyway, since all halos at a given snapshot have descs at the next?)
    '''

    #print(f"\t\t\t   Downsizing ({len(tree)} -->", end='', flush=True)

    nprogcol = tree.columns.get_loc('num_prog(4)')
    descidcol = tree.columns.get_loc('desc_id(3)')
    idcol = tree.columns.get_loc('id(1)')

    halmix = []

    atree= np.array(tree) # halwgal[n]
    roots = atree[atree[:,nprogcol]==0] # yup num_prog still at 4 
    mergers = atree[atree[:,nprogcol]>1]
    final = atree[atree[:,descidcol]=='-1']
    des = []; #pro = []; discarded = []; premerger_id = []
    finalid = final[0][idcol] # [0] because atree[row] will give list inside a list

    # if not (finalid in mergers[:,descidcol] or finalid in roots[:,descidcol]):
    #     print(f"Final halo {finalid} is not the descendent of a merger (or root)")
    #     progid = atree[np.where(finalid==atree[:,descidcol])][0][idcol]
    #     print(f"Instead, the halo it descends from is {progid}: {list(atree[np.where(finalid==atree[:,descidcol])][0])[0:7]}")
    #     if not (progid in mergers[:,descidcol] or progid in roots[:,descidcol]):
    #         print(f"This halo is also not the descendent of a merger (or root)")
    #         progid1 = atree[np.where(finalid==atree[:,descidcol])][0][idcol]
    #         print(f"Instead, the halo it descends from is {progid1}: {list(atree[np.where(progid==atree[:,descidcol])][0])[0:7]}")
    #     a=b
        
    hrc = -1; drc = -1
    if debug: print(f"\n")
    if finalid not in mergers[:,idcol]: # IF FIRST HALO (FINAL HALO) IS NOT A MERGER (ONLY 1 DESC) IT ALSO WONT BE A ROOT SO IT WILL NEVER GET ADDED... 
        halmix.append(final[0]); hrc += 1
        des.append(final[0][descidcol]); drc += 1
        if debug:
            print(f"Final halo {finalid} is not a merger, so adding it here")
            print(f"  Adding it to halmix [halmix row {hrc}]") 
            print(f"  Adding its desc id {final[0][descidcol]} to des [des row {drc}]")
    if debug: print(f"Looping through roots and mergers")
    for q, mid in enumerate(mergers[:,idcol]):
        if debug: print(f"Halo {mid} is merger {q} (descid {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]})")
        halmix.append(mergers[q]); hrc +=1 
        if debug: print(f"  Adding it to halmix [halmix row {hrc}]") 
        k=1
        descid = atree[:,descidcol][np.where(mid==atree[:,idcol])][0] # look at this halo's descendent 
        while descid not in mergers[:,idcol] and descid!='-1': # if this halos desc is not a merger or final, find the next halo who's desc is merger or final and assign it to be this halos be this halos new desc
            proid = atree[:,idcol][np.where(descid==atree[:,idcol])][0] # get the id of the prog of this halo's descendent (want to keep the progs of mergers - could this get moved outside while loop and just get it if this halo's desc is a merger?)   # added [0] 
            descid = atree[:,descidcol][np.where(descid==atree[:,idcol])][0] # look at this halo's descendent's descendent   # added [0] # desc id of halos desc  ##new descendant id where current descendant id = halo id
            k += 1
        if k>1 and proid != finalid: # if original desc of this merger was not a merger but its new desc is (as opposed to new desc being final (this is same as descid != -1, right?))
            des.append(proid); drc += 1
            halmix.append(atree[np.where(proid==atree[:,idcol])][0]); hrc +=1
            if debug:
                print(f"  Its original desc is not a merger or final, and this new desc is merger, not final")
                print(f"     Adding id of its new desc ('proid' {proid}) to des [des row {drc}]") # Add halo's new desc id at corresponding des index
                print(f"     Adding its new desc (halo at 'proid') to halmix [halmix row {hrc}]") # Add new descendent to halmix
        if descid!='-1': # if desc of [this merger]/[this mergers new desc] is not final (is a merger)
            des.append(descid); drc +=1 # Add [id of desc]/[id of new desc's desc] to des at [idx correponding to this merger]/[idx corresponding to new desc] # removed [0]
            if debug: print(f"  Adding {'new' if k>1 else 'original'} descid {descid} to des [des row {drc}]") 
        else: # descid==-1, e.g. either didnt enter while loop or entered and found final
            if k == 1:
                des.append(atree[:,descidcol][np.where(mid==atree[:,idcol])][0]); drc += 1 # ADD ORIGINAL DESC??? THAT ONLY WORKS IF DIDNT ENTER WHILE (E.G. PROG OF FINAL IS MERGER, AND WE ARE ON THAT MERGER)
                if debug: print(f"  Adding original desc id {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]} to des [des row {drc}]")
            else: 
                des.append(finalid); drc += 1 # Add id of new desc's desc (final) to des at idx corresponding to new desc
                if debug: print(f"  Adding new descid (final halo {proid} to des [des row {drc}]")
                if proid != finalid: print(f"Error: entered loop and found final, but proid {proid} != finalid {finalid}"); a=b
        
        if hrc != drc: print(f"Error: hrc {hrc} != drc {drc}"); a=b
        if len(halmix[-1]) != len(list(tree.columns)):
            print(f"\n[haloset {num}] Error at halmix row {hrc} (after proccessing merger {q}) has length {len(halmix[-1])} not n feats {len(list(tree.columns))} \n\thalmix[-1] {halmix[-1]}"); a=b

    c = 0     
    for r in roots:
        #print(f"Halo {r[idcol]} is root {c} (descid {atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0]})")
        halmix.append(r); hrc +=1 
        #print(f"  Adding it to halmix [halmix row {hrc}]")
        descid=atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0] 
        k=1
        while descid not in mergers[:,idcol] and descid!='-1': # could remove descid!=-1 right? root should never go directly to final
            proid=atree[:,idcol][np.where(descid==atree[:,idcol])][0] 
            descid=atree[:,descidcol][np.where(descid==atree[:,idcol])][0]
            k+=1
        if k>1 and proid!=finalid: # if it entered the above while loop (desc of this merger is not a merger?) and progenitor is not the final halo (?)
            halmix.append(atree[np.where(proid==atree[:,idcol])][0]); hrc +=1 
            des.append(proid); drc += 1
            if debug:
                print(f"  Its original desc is not a merger or final, and new proid is also not final")
                print(f"     Adding halo at new 'proid' to halmix [halmix row {hrc}]")
                print(f"     Adding new 'proid' {proid} to des [des row {drc}]")
        if descid!='-1':
            des.append(descid); drc +=1 
            if debug: print(f"  Adding {'new' if k>1 else 'original'} {descid} to des [des row {drc}]")
        else: # descid==-1, e.g. didn't enter loop (dont need to worry about finding final as next desc of a root)
            des.append(atree[:,descidcol][np.where(r[idcol]==atree[:,idcol])][0]); drc += 1
            if debug: print(f"  Adding original desc id {atree[:,descidcol][np.where(mid==atree[:,idcol])][0]} to des [des row {drc}]") 

        if hrc != drc: print(f"Error: hrc {hrc} != drc {drc}"); a=b
        if len(halmix[-1]) != len(list(tree.columns)):
            print(f"\n[haloset {num}] Error: row at at halmic row {hrc} (after proccessing root {c}) has length {len(halmix[-1])} not n feats {len(list(tree.columns))} \n\thalmix[-1] {halmix[-1]}"); a=b

        c += 1
    
    tree_out = pd.DataFrame(halmix, columns=tree.columns)
    tree_out['newdesc_id'] = des

    for i in range(len(tree_out)):
        if not (tree_out.iloc[i]['newdesc_id'] in list(tree_out['id(1)']) or tree_out.iloc[i]['newdesc_id'] == '-1'): # this halos original descendent was cut without updateing this halos descendent 
            print(f"\n[haloset {num}] Error at row {i}: no halos in the tree have this halo's descdendent ({tree_out.iloc[i]['newdesc_id']})");
            print(f"\nhalo: {list(tree_out.iloc[i])}")
            print(f"\tlist(tree_out['id(1)'])[0:5]: {list(tree_out['id(1)'])[0:5]}")
            print(f"\tlist(tree_out['newdesc_id'])[0:5]: {list(tree_out['newdesc_id'])[0:5]}")
            a=b

    try: 
        np.array(tree_out, dtype=float)
    except: 
        print(f"[haloset {num}] Cant set tree_out to array")
        for i in range(len(halmix)):
            try: np.array(tree_out.iloc[i], dtype=float)
            except: 
                print(f"[haloset {num}] Error at row {i}")
                print(f"halmix[i] \n{halmix[i]} \nhalmix[i+1] \n{halmix[i+1]}")
                print(f"tree_out.iloc[i] \n{list(tree_out.iloc[i])} \ntree_out.iloc[i+1] \n{list(tree_out.iloc[i+1])}")
                a=b

    #print(f"{len(halmix)})", flush=True) 

    return tree_out


###################
# Debuging functions
###################

def prep_mstar(obscat_path, volname, save_path):
    
    # Load files for crossmatching to rstar IDs
    crossmatchRSSF_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volname}_DM/postprocessing/trees/rockstar/matches/rockstar_subhalo_matching_to_FP.hdf5'
    cmRSSF_list = h5py.File(crossmatchRSSF_path, 'r')['Snapshot_99']['SubhaloIndexDarkRockstar_SubLink'][:]
    rstar_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/{volname}_DM/postprocessing/trees/rockstar/groups_099/fof_subhalo_tab_099.0.hdf5'
    rstar_subhalo_ids = np.array(h5py.File(rstar_path, 'r')['Subhalo']['Subhalo_ID'])

    # Load subfind data 
    subhalos = il.groupcat.loadSubhalos(basePath = obscat_path, snapNum=99)
    sh_mstar = subhalos['SubhaloMassType'][:,4]

    # Add mstar data to dict
    all_mstar = {}
    for idx in range(len(sh_mstar)):
        # Get rstar_id for matching to tree
        subfind_read_idx =  idx
        rstar_read_idx = cmRSSF_list[subfind_read_idx]
        rstar_id = int(rstar_subhalo_ids[rstar_read_idx]) 
        # Add m_star to dict under rstar_id
        all_mstar[str(rstar_id)] = sh_mstar[idx] 
    
    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(all_mstar, f) 

def prep_mhaloz0(ctrees_path, mstar_ids, save_path):

    # Load ctrees data (just one file)
    all_names = ['#scale(0)', 'id(1)', 'desc_scale(2)', 'desc_id(3)', 'num_prog(4)', 'pid(5)', 'upid(6)', 'desc_pid(7)', 'phantom(8)', 'sam_Mvir(9)', 'Mvir(10)', 'Rvir(11)', 'rs(12)', 'vrms(13)', 'mmp?(14)', 'scale_of_last_MM(15)', 'vmax(16)', 'x(17)', 'y(18)', 'z(19)', 'vx(20)', 'vy(21)', 'vz(22)', 'Jx(23)', 'Jy(24)', 'Jz(25)', 'Spin(26)', 'Breadth_first_ID(27)', 'Depth_first_ID(28)', 'Tree_root_ID(29)', 'Orig_halo_ID(30)', 'Snap_idx(31)', 'Next_coprogenitor_depthfirst_ID(32)', 'Last_progenitor_depthfirst_ID(33)', 'Last_mainleaf_depthfirst_ID(34)', 'Tidal_Force(35)', 'Tidal_ID(36)', 'Rs_Klypin', 'Mvir_all', 'M200b', 'M200c', 'M500c', 'M2500c', 'Xoff', 'Voff', 'Spin_Bullock', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]', 'A[z]', 'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', 'A[y](500c)', 'A[z](500c)', 'T/|U|', 'M_pe_Behroozi', 'M_pe_Diemer', 'Halfmass_Radius']
    loadcols = [all_names.index(name) for name in ['#scale(0)', 'desc_scale(2)', 'num_prog(4)', 'phantom(8)', 'Mvir(10)', 'id(1)', 'desc_id(3)', 'pid(5)', 'Orig_halo_ID(30)']]
    pd0 = pd.read_table(f'{ctrees_path}/tree_0_0_0.dat', header=0, skiprows=0, delimiter='\s+', usecols=loadcols)
    raw = pd0.drop(axis=0, index=np.arange(48))
    halos = raw[~raw.isna()['desc_id(3)']] # rows (nodes)
    indices = [i for i, x in enumerate(halos['desc_id(3)']) if x == -1] # Indices where desc_id is -1 (first line of each tree)
    split_halos = np.split(np.array(halos), np.array(indices)) # list of 2d tree arrays. NOTE: NOT HAVING TO REMOVE FIRST SPLIT - WHY DID CHRISTIAN?? 

    # Add mhalo data to dict
    all_mhaloz0 = {}
    for i in range(len(split_halos)): 
        rstar_id = str(int(list(split_halos[i][0])[8])) # CHANGE: ADD INT AND STR # same as: haloset = split_halos[i]; tree = pd.DataFrame(data = haloset, columns = halos.columns); halo_z0 = tree.iloc[0]; rstar_id = halo_z0['Orig_halo_ID(30)']
        if rstar_id not in mstar_ids: 
            continue
        m_haloz0 = list(split_halos[i][0])[7] # same as: m_halo_z0 = halo_z0['Mvir(10)']
        all_mhaloz0[rstar_id] = m_haloz0
    
    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(all_mhaloz0, f)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def describe_dataset(datapath, dataset):
          
    data_meta = json.load(open(f'{datapath}{dataset}_meta.json', 'rb'))
    obs_meta = data_meta[1]['obs meta']
    tree_meta = data_meta[2]['tree meta']
    print(f"Dataset: {dataset}")
    print(f"Cuts: sizelim = {tree_meta['sizelim']}, Mlim = {tree_meta['Mlim']}, reslim = {obs_meta['reslim']}")
    print(f"{obs_meta['tot galaxies']} total galaxies in phot catalog, {obs_meta['tot galaxies saved']} saved")
    print(f"      {obs_meta['galaxies res cut']} too low res, {obs_meta['galaxies sat cut']} not central ")
    print(f"{tree_meta['tot trees']} total trees in ctrees files, {tree_meta['tot trees saved']} saved")
    print(f"      {tree_meta['trees mass cut']} with log(M) < {tree_meta['Mlim']}, {tree_meta['trees sat cut']} not central, {tree_meta['trees no mergers cut']} without mergers, {tree_meta['trees size cut']} with num halos > {tree_meta['sizelim']}")

#############################
# Loading data for train/test 
#############################

def create_subset(data_params, return_set): 
    '''
    Load data from pickle, transform, remove unwanted features and targets, and split into train and val or train and test
    NOTE: For graph data like this need to apply fitted transformers to each graph individually (cant just stack like for images), but want to fit to train and val seperately
          Cant think of a better way than looping through data twice (once to fit transformer, once to transform)
          Chrisitan just used the same transfomer on train and test i think (probabaly ok if that transformer was just fitted on train or trainval?)
    '''

    # Use dataset meta file to see names of all feats and targs in dataset
    data_path = data_params['data_path']
    data_file = data_params['data_file']
    data_meta_file = osp.expanduser(osp.join(data_path, data_file.replace('.pkl', '_meta.json')))
    data_meta = json.load(open(data_meta_file, 'rb'))
    all_labelnames = data_meta[1]['obs meta']['label_names'] 
    all_featnames = data_meta[2]['tree meta']['featnames']

    # Load dataset subset parameters 
    use_targidxs = [all_labelnames.index(targ) for targ in data_params['use_targs']] # targets = data_params['targets']
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in data_params['use_feats']] #  # have use_feats be names not idxs! in case we want to train on fewer features without recreating dataset
    transfname = data_params['transf_name']
    splits = data_params['splits']

    # Load data from pickle
    print(f'Dataset subset doesnt yet exist. Creating from {data_path}{data_file}', flush=True)
    data = pickle.load(open(osp.expanduser(f'{data_path}/{data_file}'), 'rb'))

    # Split into trainval and test
    testidx_file = osp.expanduser(f'{data_path}{data_file.replace(".pkl", "")}_testidx{splits[2]}.npy')
    if not os.path.exists(testidx_file): 
        print(f'   Test indices for {splits[2]}% test set not yet saved for this dataset ({data_file}). Creating and saving in {testidx_file}', flush=True)
        testidx = np.random.choice(np.linspace(0, len(data)-1, len(data), dtype=int), int(len(data)*(splits[2]/100)), replace=False)
        np.save(testidx_file, testidx)
    else: 
        print(f'   Loading test indices from {testidx_file}', flush=True)
        testidx = np.array(np.load(testidx_file))
    test_data = []
    trainval_data = [] 
    for i, d in enumerate(data):
        if i in testidx:
            test_data.append(d)
        else:
            trainval_data.append(d)
                
    # Fit transformers to all trainval graph data and all test graph data. Note: Use all cols for transform 
    if transfname != "None":
        
        traintransformer_path = osp.expanduser(f'{data_path}{data_file.replace(".pkl", "_"+transfname+"_train.pkl")}')
        if not os.path.exists(traintransformer_path):
            print(f'   Fitting transformer to train/val data (will save in {traintransformer_path})', flush=True)
            ft_train = run_utils.fit_transformer(trainval_data, all_featnames, save_path=traintransformer_path, transfname=data_params['transf_name']) # maybe should do seperate for train/val too, but as long as not fitting on final test its probabaly ok
        else: 
            print(f'   Applying transformer to train/val data (loaded from {traintransformer_path})', flush=True)
            ft_train = pickle.load(open(traintransformer_path, 'rb')) 
        
        testtransformer_path = osp.expanduser(f'{data_path}{data_file.replace(".pkl", "_"+transfname+"_test.pkl")}')
        if not os.path.exists(testtransformer_path):
            print(f'   Fitting transformer to test data (will save in {testtransformer_path})', flush=True)
            ft_test = run_utils.fit_transformer(test_data, all_featnames, save_path=testtransformer_path, transfname=data_params['transf_name']) 
        else: 
            print(f'   Applying transformer to test data (loaded from {testtransformer_path})', flush=True)
            ft_test = pickle.load(open(testtransformer_path, 'rb')) 

    # Transform each graph, and remove unwanted features and targets
    print(f"   Selecting desired features {data_params['use_feats']} and targets {data_params['use_targs']}", flush=True)
    trainval_out = []
    test_out = []
    test_z0x_all = []
    for graph in trainval_data:
        x = graph.x
        if transfname != "None": 
            x = ft_train.transform(graph.x)
        x = torch.tensor(x[:, use_featidxs])
        trainval_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y[use_targidxs]))
    for graph in test_data:
        x = graph.x
        test_z0x_all.append(x[0]) # add all feats from final halo
        if transfname != "None": 
            x = ft_test.transform(x)
        x = torch.tensor(x[:, use_featidxs])
        test_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y[use_targidxs]))

    # Split trainval into train and val
    print(f"   Splitting train/val into train and val", flush=True)
    val_pct = splits[1]/(splits[0]+splits[1]) # pct of train
    train_out = trainval_out[:int(len(trainval_out)*(1-val_pct))]
    val_out = trainval_out[int(len(trainval_out)*(val_pct)):]

    # Save (seperately for train and test to aviod any mistakes!)
    tag = f"_{use_featidxs}_{use_targidxs}_{transfname}_{splits}"
    train_path = osp.expanduser(f'{data_path}{data_file.replace(".pkl", tag)}_train.pkl')
    test_path = osp.expanduser(f'{data_path}{data_file.replace(".pkl", tag)}_test.pkl')
    print(f"   Saving train+val sub-dataset as {train_path}", flush=True)
    print(f"   Saving test sub-dataset as {test_path}", flush=True)
    pickle.dump((train_out, val_out), open(train_path, 'wb'))
    pickle.dump((test_out, test_z0x_all), open(test_path, 'wb')) # also saving z0 feats for all test halos, for future exploration

    # If "test" run return train+val and test (NOTE: Christian uses 'test' to mean training the same model again completely from scratch on train+val, then testing on test)
    if return_set == "test":
        return trainval_out, test_out, test_z0x_all 
    if return_set == 'train':
        return train_out, val_out 
    
def get_subset_path(data_params, set):

    # Get subset parameters
    data_path = data_params['data_path']
    data_file = data_params['data_file']
    data_meta_file = osp.expanduser(osp.join(data_path, data_file.replace('.pkl', '_meta.json')))
    data_meta = json.load(open(data_meta_file, 'rb'))
    all_label_names = data_meta[1]['obs meta']['label_names'] 
    all_featnames = data_meta[2]['tree meta']['featnames']
    use_targidxs = [all_label_names.index(targ) for targ in data_params['use_targs']] # targets = data_params['targets']
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in data_params['use_feats']] #  # have use_feats be names not idxs! in case we want to train on fewer features without recreating dataset
    transfname = data_params['transf_name']
    splits = data_params['splits']
    tag = f"_{use_featidxs}_{use_targidxs}_{transfname}_{splits}"

    # Return train or test path
    if set =='test':
        return osp.expanduser(f'{data_path}{data_file.replace(".pkl", tag)}_test.pkl')  
    else:
        return osp.expanduser(f'{data_path}{data_file.replace(".pkl", tag)}_train.pkl')  



#############################
# For just final halos model
#############################
    
# Define Data class for ease of access 
class SimpleData(Dataset_notgeom):

  def __init__(self, X, Y):
    self.X = X
    self.y = Y
    self.len = len(X)
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
  def __len__(self):
    return self.len

def data_finalhalos(allgraphs, use_featidxs, use_targidxs, ft_train): 
    
    X = []; Y = []
    for graph in allgraphs:
        x = np.expand_dims(graph.x[0].detach().numpy(), axis=0)
        x = ft_train.transform(x).squeeze()
        X.append(x[use_featidxs].astype(np.float32))
        Y.append(graph.y[use_targidxs].detach().numpy().astype(np.float32))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    traindata = SimpleData(X_train, Y_train)
    testdata = SimpleData(X_test, Y_test)
  
    return traindata, testdata

def data_fake(n=20000): 
    
    X = []; Y = []
    for i in range(n):
        x = np.random.rand(5)
        X.append(x.astype(np.float32))
        Y.append(np.array([np.mean(x)+np.random.randint(-100, 100)/10000, np.std(x)+np.random.randint(-100, 100)/10000]).astype(np.float32))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    traindata = SimpleData(X_train, Y_train)
    testdata = SimpleData(X_test, Y_test)
  
    return traindata, testdata

def propsdata_finalhalos(props_data, use_featidxs, use_targidxs):

    X = []; Y = []
    for graph in props_data:
        X.append(graph.x[0].detach().numpy()[use_featidxs].astype(np.float32))
        Y.append(graph.y[use_targidxs].detach().numpy().astype(np.float32))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    traindata = SimpleData(X_train, Y_train)
    testdata = SimpleData(X_test, Y_test)
  
    return traindata, testdata
