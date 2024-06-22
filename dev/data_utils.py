
import pickle, json, time, os, sys, io
import numpy as np
import os.path as osp
from datetime import date, datetime

import torch
os.environ['TORCH'] = torch.__version__
import torch_geometric as tg
from torch_geometric.data import Data
from torch.utils.data import Dataset as Dataset_notgeom
import networkx as nx
from collections import defaultdict
import h5py

import illustris_python as il
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import sklearn.metrics as skmetrics
import multiprocessing as mp
try:
    import run_utils
except ModuleNotFoundError:
    try: 
        from dev import run_utils
    except ModuleNotFoundError:
        try:
            sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
            from ObservablesFromTrees.dev import run_utils
        except ModuleNotFoundError:
            sys.path.insert(0, 'ObservablesFromTrees/dev/')
            from dev import run_utils

# Make dictionary of (cm_ID, tree DF) for all trees
def prep_trees(ctrees_path, featnames, phot_ids, metafile, savefile, zcut=('before', np.inf), Mlim=10, sizelim=np.inf, tinytest=False, downsize_method='', multi=False, check_gain_with='prog'): 
    '''
    PROPBABLY SHOULD UPDATE THIS TO:
        1. Store 
    '''
    
    
    # Prepare to load tree data 
    tree_path = osp.expanduser(ctrees_path)
    #loadcols = [allnames.index(name) for name in allnames if name not in ignorenames] # col idxs to load (remove ignorenames)
    print(f'\tTree features to include: {featnames}', flush=True)
    
    # Initialize meta dict
    meta = {'Mlim': str(Mlim), 'sizelim': str(sizelim), 'zcut': str(zcut), 
            'downsize method': downsize_method,
            'pct gain to include': 10,
            'check_gain_with': check_gain_with,
            'lognames': ['Mvir(10)', 'Mvir_all', 'M200b', 'M200c', 'M2500c', 'M_pe_Behroozi', 'M_pe_Diemer'], 
            'minmaxnames': ['Jx(23)', 'Jy(24)', 'Jz(25)'],
            'featnames': featnames}  #if not add_gain else featnames + ['gainPct'] # Add 'gainPct' to feat names if adding that
    
    # Load files
    t0_all = time.time()
    treefiles = [f for f in os.listdir(tree_path) if f.endswith('.dat') and f.startswith('tree_')]
    if tinytest: treefiles=treefiles[0] # for small set to test on, just use trees in one file
    
    # Process trees in each file, breaking files into sets if multithreading
    if downsize_method not in [1, 3, 4, 7, 'None']: raise ValueError(f'Invalid downsize method {downsize_method}')
    if not multi: 
        all_trees, processing_meta = process_treefiles(treefiles, tree_path, featnames, meta['lognames'], meta['minmaxnames'], zcut, Mlim, sizelim, downsize_method, phot_ids, check_gain_with, save_path=savefile)
    else:    
        procs_outdir = savefile.replace('_alltrees.pkl', '_procs_temp/')
        print(f"Creating temporary output directory {procs_outdir} for each thread to store tree and meta files.")
        os.makedirs(procs_outdir)               
        n_threads = 10 #os.cpu_count() # On Popeye, each node has 48 cores. So (assuming only requesting 1 node) os.cpu_count() is 48. 
        filesets = np.array_split(treefiles, n_threads) # it wouldnt really help to split into more sets then there are cpus, right?
        print(f"Splitting the {len(treefiles)} CT files into {len(filesets)} sets of ~{len(filesets[0])} files each, to be processed by {n_threads} threads.")
        pool = mp.Pool(processes=n_threads) 
        results = []
        for i in range(len(filesets)):
            args = (filesets[i], tree_path, featnames, meta['lognames'], meta['minmaxnames'], zcut, Mlim, sizelim, downsize_method, phot_ids, check_gain_with, procs_outdir, i)
            result = pool.apply_async(process_treefiles, args=args)#), callback=response)
            results.append(result.get())
        pool.close()
        pool.join()
        for res in results:
            print(f"{res}")
        if len(os.listdir(procs_outdir)) == 0: 
            raise ValueError('All procs finished but no files saved in procs_outdir')
        all_trees, processing_meta = collect_proc_outputs(procs_outdir, task='trees') 
 
    # Save full alltrees
    pickle.dump(all_trees, open(savefile, 'wb')) # savefile = name of file containing collected trees from all file sets
    with open(metafile, 'r+') as f: 
        listmeta = json.load(f)
        
    # Update and save full meta
    meta.update(processing_meta) # add keys and values from processing_meta to meta
    meta['n_threads'] = n_threads if multi else 'NaN'
    meta['Total time'] = np.round(time.time() - t0_all, 4)
    if f'tree meta' in [list(listmeta[i].keys())[0] for i in range(len(listmeta))]: # If targ_type meta already exists (this is a re-do), remove old targ_type meta
        rem_idx = np.where(np.array([list(listmeta[i].keys())[0] for i in range(len(listmeta))]) == f'tree meta')[0][0]
        listmeta.pop(rem_idx) 
    listmeta.append({'tree meta':meta})  
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)
    f.close()
    
def process_treefiles(fileset, tree_path, featnames, logcols, minmaxcols, zcut, Mlim, sizelim, downsize_method, phot_ids, check_gain_with, save_path, proc_id=''):

    # Create dicts to store trees and meta results from this file set [note that file set is all tree files if not multithreading]
    all_trees_set = {}
    set_meta = {'tot trees': 0, 'tot trees saved': 0, 'trees mass cut': 0, 'trees sat cut': 0, 'trees size cut': 0, 'trees no mergers cut': 0}
    if downsize_method in [4,7]: set_meta['trees affected by mass keep'] = 0

    # Set file names for set output
    if proc_id != '':
        savefile_set = f'{save_path}/{proc_id}_alltrees.pkl' # save_path is ../Data/{dataset}_procs_temp/
        metafile_set = f'{save_path}/{proc_id}_treemeta.json'
    else:
        savefile_set = save_path # save_path is ../Data/{dataset}_allobs.pkl
   
    # Loop through file set
    proc_name = f"[Proccess {proc_id}]" if proc_id != '' else ''
    print(f'\t{proc_name} Begining iteration through {len(fileset)} files', flush=True)
    downsize_func = run_utils.get_downsize_func(downsize_method)
    fcount = 0
    t0 = time.time()
    for treefile in fileset: 
                
        # Load tree sets (each file has many trees, with a tree being a header line followed by rows of nodes)
        #print(f'\t{proc_name} Loading trees from {treefile} ({fcount} of {len(fileset)})', flush=True)
        t0_file = time.time()
        pd1 = pd.read_table(f'{tree_path}/{treefile}', skiprows=0, delimiter='\s+', dtype=str) # read ALL COLS in (as strings) 
        raw = pd1.drop(axis=0, index=np.arange(48)) # remove rows 0 to 48, which are one large set of comments
        del pd1
        print(f'\t{proc_name} Loaded trees from {treefile} ({fcount} of {len(fileset)})', flush=True)
        
        # Split into df of just trees by detecting the tree header lines (they have NaN in all but the first two cols)
        halos = raw[~raw.isna()['desc_id(3)']] # rows (nodes)
        del raw
        
        # For all halos in this file, change dtypes, make z cut, scale columns
        halos = change_dtypes(halos, featnames) # only change for features (keep ids as strings)
        halos = make_zcut(halos, zcut)
        halos = scale(halos, featnames, logcols, minmaxcols) # WHY WAS TYPO HERE CUASING MULTITHREADING SCRIPT TO JUST SKIP??? (typo was forgetting to remove extra return, halos, set_meta = scale(..))    #REMOVED num_prog FROM COLS GETTING LOG SCALED 

        # For each tree, construct dataframe, downsize, save to tree dict with corresponding crossmatch ID as key
        indices = [i for i, x in enumerate(halos['desc_id(3)']) if x == '-1'] # (3/5) THIS WAS NOT '-1' BEFORE! Indices where desc_id is -1 (first line of each tree)
        split_halos = np.split(np.array(halos), np.array(indices[1:])) # list of 2d tree arrays #print(f'\t\tSplitting into {len(split_halos)} individual trees', flush=True)
        set_meta['tot trees'] += len(split_halos)
        print(f"\t{proc_name} {treefile} has {len(split_halos)} trees", flush=True)
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
            toolong = len(haloset) > float(sizelim) # Christian cut out those over 20000, but Rachel says not to
            # nomergers = len(haloset[haloset[:,4]>1]) == 0
            if (toolong or lowmass or satelite or SFcut): # or nomergers): 
                if lowmass: set_meta['trees mass cut'] += 1
                if satelite: set_meta['trees sat cut'] += 1
                if toolong: set_meta['trees size cut'] += 1
                continue
            # print(f'\t\t\tProcessing haloset {i} into tree', flush=True)
            tcount += 1
            # Remove "unimportant" subhalos and add newdesc_id with updated descendent
            print(f"\t{proc_name} Downsizeing tree {i} from file {treefile}", flush=True)
            if downsize_method in [4,7]: 
                dtree, somemasskept = downsize_func(tree, check_gain_with, num=i)
                if somemasskept: set_meta['trees affected by mass keep'] += 1
            elif downsize_method in [1,3]: dtree = downsize_func(tree, num=i)
            elif downsize_method == 'None': 
                dtree = tree
                dtree['newdesc_id'] = dtree['desc_id(3)']
            else: raise ValueError('Invalid downsize method')
            dtree = dtree.rename(columns={'desc_id(3)': 'olddesc_id'}) # rename desc_id to olddesc_id
            # Add (rstar_id , tree) to dicts
            all_trees_set[rstar_id] = dtree
        
        set_meta['tot trees saved'] += tcount
        fcount+=1

        print(f"\t{proc_name} Done with {treefile} ({fcount} of {len(fileset)}). Took {np.round(time.time()-t0_file, 4)}s. Processed {tcount} of {len(split_halos)} trees", flush=True)
    
    set_meta['total time'] = np.round(time.time() - t0, 4)
    print(f"\t{proc_name} Done with all files. Took {np.round(time.time() - t0, 4)} s to save {set_meta['tot trees saved']} total trees", flush=True)
    
    if proc_id != '': # If this is a thread, save set results. If not, just return them. 
        
        print(f'\t{proc_name} Saving trees in {savefile_set} and saving processing meta to {metafile_set}', flush=True) 
        pickle.dump(all_trees_set, open(savefile_set, 'wb'))
        pickle.dump(set_meta, open(metafile_set, 'wb'))

        return [proc_id, set_meta]
    
    else:
        return all_trees_set, set_meta

def collect_proc_outputs(procs_outdir, task=''):

    if task not in ['trees', 'photdat', 'propsdat']: raise ValueError(f"Multithreading only done for creating trees and final dat objects")
    if task == 'trees': 
        print(f"Collecting trees and meta from all threads in {procs_outdir}", flush=True)
        processing_meta = {'tot trees': 0, 'tot trees saved': 0, 'trees mass cut': 0, 'trees sat cut': 0, 'trees size cut': 0, 'trees no mergers cut': 0, 'trees affected by mass keep': 0}
        all = {}
        name = 'alltrees'
    if 'dat' in task: 
        print(f"Collecting data objects from all threads in {procs_outdir}", flush=True)
        processing_meta = {'num graphs': 0}
        all = []
        name = task

    # Append all set results 
    set_files = [file for file in os.listdir(procs_outdir) if file.endswith(f'{name}.pkl')] # _alltrees.pkl, _photdat.pkl, or _propsdat.pkl
    print(f"{len(set_files)} {name} files to collect", flush=True)
    for f in set_files:
        all_set = pickle.load(open(osp.join(procs_outdir, f), 'rb'))
        if task == 'trees': all.update(all_set)
        else: 
            for graph in all_set: 
                all.append(graph)

    # Add up total counts and store in meta 
    set_metafiles = [file for file in os.listdir(procs_outdir) if file.endswith(f'{task}meta.json')]
    print(f"{len(set_metafiles)} {task} meta files to collect", flush=True)
    for f in set_metafiles:
        set_meta = pickle.load(open(procs_outdir+f, 'rb'))
        for key in processing_meta.keys():
            processing_meta[key] += set_meta[key]

    return all, processing_meta

# Make dictionary of (subhalo, obs) for all central subhalos
def prep_targs(obscat_path, crossmatchRSSF_path, rstar_path, metafile, savefile, reslim=100, targ_type='phot'):
    '''
    PROBABLY SHOULD UPDATE THIS DO STORE BOTH PHOT AND PROPS AT THE SAME TIME NEXT TIME I RUN A NEW DS
    '''
    # Should I store mstar and res as well? 
    #   Would it make later access easier?
    #   Won't help for looking at other targs, and would still need to re-map to trees, since dont want graphs to include things other)

    if targ_type == 'phot':
        label_names = ["U", "B", "V", "K", "g", "r", "i", "z"]
        to_store = 'phot'
    if targ_type == 'props':
        label_names = ['SubhaloBHMass', 'SubhaloBHMdot','SubhaloGasMetallicity','SubhaloHalfmassRad','SubhaloMass','SubhaloGasMass','SubhaloStelMass','SubhaloSFR','SubhaloStarMetallicity','SubhaloVelDisp','SubhaloVmax','SubhaloRes']
        to_store = label_names

    meta = {'label_names': label_names, 'reslim': reslim, 
            'tot galaxies': 0, 'tot galaxies saved': 0, 
            'galaxies res cut': 0, 'galaxies sat cut': 0, 
            'start stamp': str(datetime.now())}

    # Collect phot or props
    all_obs, meta = collect_targs(to_store, crossmatchRSSF_path, rstar_path, obscat_path, reslim, meta)

    # Save to pickle
    print(f'\tSaving {targ_type} in {savefile} and meta in {metafile}', flush=True)     
    with open(savefile, 'wb') as f:
        pickle.dump(all_obs, f)
    with open(metafile, 'r+') as f: 
        listmeta = json.load(f) 
    if f'{targ_type} meta' in [list(listmeta[i].keys())[0] for i in range(len(listmeta))]: #list(listmeta[-1].keys())[0] == f'{targ_type} meta': # If targ_type meta already exists (this is a re-do), remove old targ_type meta
        rem_idx = np.where(np.array([list(listmeta[i].keys())[0] for i in range(len(listmeta))]) == f'{targ_type} meta')[0][0]
        listmeta.pop(rem_idx) #listmeta[:-1]
    listmeta.append({f'{targ_type} meta': meta}) 
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)
    f.close()
    
    return all_obs     


def collect_targs(to_store, crossmatchRSSF_path, rstar_path, obscat_path, reslim, meta=None):
    
    # Load files for crossmatching to rstar IDs
    cmRSSF_list = h5py.File(crossmatchRSSF_path, 'r')['Snapshot_99']['SubhaloIndexDarkRockstar_SubLink'][:]
    rstar_subhalo_ids = np.array(h5py.File(rstar_path, 'r')['Subhalo']['Subhalo_ID'])

    # Load subhalo photometry and resolution
    subhalos = il.groupcat.loadSubhalos(basePath = obscat_path, snapNum=99)
    sh_phot = subhalos['SubhaloStellarPhotometrics']
    sh_res = subhalos['SubhaloLenType'][:,4] # 4 for star particles 
    if meta!= None: meta['tot galaxies'] = len(sh_phot)

    # Get central subhalo indices
    groups = il.groupcat.loadHalos(basePath = obscat_path, snapNum=99)
    ctl_idxs = groups['GroupFirstSub'] # for each group, the idx w/in "subhalos" that is its central subhalo (-1 if group as no subhalos)
    ctl_idxs = ctl_idxs[ctl_idxs != -1] # remove groups with no subhalos
    if meta!= None: meta['galaxies sat cut'] = len(sh_phot) - len(ctl_idxs)

    # Add photometry for all central subhalos to dict
    all_targs = {} 
    for idx in ctl_idxs:
        # Remove if not central or resolution too low
        if (sh_res[idx] < reslim): 
            if meta!= None: meta['galaxies res cut'] += 1
            continue
        # Get rstar_id for matching to trees
        subfind_read_idx =  idx
        rstar_read_idx = cmRSSF_list[subfind_read_idx]
        rstar_id = int(rstar_subhalo_ids[rstar_read_idx])
        # Add obs to dict under rstar_id       
        if to_store == 'phot':
            all_targs[str(rstar_id)] = sh_phot[idx] 
        else:
            info = []
            for targ in to_store:
                if targ == 'SubhaloRes': info.append(sh_res[idx])
                elif targ == 'SubhaloGasMass': info.append(subhalos['SubhaloMassType'][idx,0]) # WAS [IDX][O] BUT I DONT THINK THAT SHOULD BE THE PROBLEM
                elif targ == 'SubhaloStelMass': info.append(subhalos['SubhaloMassType'][idx,4])
                else: info.append(subhalos[targ][idx])          
            all_targs[str(rstar_id)] = info # all_targs.loc[len(all_targs)] = info
        if meta!= None: meta['tot galaxies saved'] += 1

    if meta == None:
        return all_targs
    return all_targs, meta

# Combine prepared trees and TNG subfind targets into pytorch data objects
def make_final_products(alltrees, allobs, featnames, metafile, savefile, multi=False): 
    '''
    Combine trees and corresponding targets into graph data objects
    For efficiency, coult try to avoid creating edges again when adding graphs for a new targ type for a dataset tha already has graphs of another type 
        Instead, check if final products already exist for phot, and then use the x and edge data for those. 
        (Really what would be most logical would be to have alltrees be the graphs except without y, and then here just add y to each graph for whatever targ)
    '''

    print(f'Combining trees and targets ({len(list(alltrees.keys()))} total) into full data object', flush=True)
    meta = {'num graphs':0, 'start stamp': str(datetime.now())}
    targ_type = 'props' if 'props' in savefile else 'phot' if 'phot' in savefile else 'err'

    # Process
    if not multi: 
        dat, processing_meta = process_graphs(alltrees, allobs, featnames, savefile, targ_type, proc_id='')
    else:    
        procs_outdir = savefile.replace(f'_{targ_type}_final.pkl', '_procs_temp/') # use same procs temp dir as for tree creation
        if not os.path.exists(procs_outdir): 
            print(f"Temporary thread output directory {procs_outdir} does not already exists (trees were created without multithreading?), so creating.")
            os.makedirs(procs_outdir)               
        n_threads = 10 #os.cpu_count() # On Popeye, each node has 48 cores. So (assuming only requesting 1 node) os.cpu_count() is 48. 
        keysets = np.array_split(list(alltrees.keys()), n_threads)
        print(f"Splitting {len(alltrees)} trees into {len(keysets)} sets ({[len(ks) for ks in keysets]}) trees each, to be processed by {n_threads} threads.")
        pool = mp.Pool(processes=n_threads) 
        results = []
        for i in range(len(keysets)):
            keys = keysets[i]
            treeset = {k: alltrees[k] for k in keys}
            args = (treeset, allobs, featnames, procs_outdir, targ_type, i)
            result = pool.apply_async(process_graphs, args=args)#), callback=response)
            results.append(result.get())
        pool.close()
        pool.join()
        for res in results:
            print(f"{res}")
        if len(os.listdir(procs_outdir)) == 0: raise ValueError('All procs finished but no files saved in procs_outdir')
        dat, processing_meta = collect_proc_outputs(procs_outdir, task=f'{targ_type}dat') 
                
    # Save pickled dataset
    print(f'Saving dataset to {savefile} and adding meta to {metafile}', flush=True) 
    with open(osp.expanduser(savefile), 'wb') as f:
        pickle.dump(dat, f)
    
    # Checks
    if len(dat) != len(alltrees): raise ValueError(f"Number of graphs ({len(dat)}) does not match number of trees ({len(alltrees)})")

    # Save meta
    meta.update(processing_meta) # add keys and values from processing_meta to meta
    with open(metafile, 'r+') as f: 
        listmeta = json.load(f) 
    if f'final {targ_type} meta' in [list(listmeta[i].keys())[0] for i in range(len(listmeta))]: # If targ_type meta already exists (this is a re-do), remove old targ_type meta
        rem_idx = np.where(np.array([list(listmeta[i].keys())[0] for i in range(len(listmeta))]) == f'final {targ_type} meta')[0][0]
        listmeta.pop(rem_idx) #listmeta[:-1]
    listmeta.append({f'final {targ_type} meta':meta}) 
    with open(metafile, 'w') as f:
        json.dump(listmeta, f)
    f.close()
    # if multi: os.rmtree(procs_outdir) # remove temp files
            
    return dat

def process_graphs(alltrees_set, allobs, featnames, save_path, targ_type, proc_id=''):

    # Create objetcs to store graphs and meta results from this file set [note that tree set is all trees if not multithreading]
    set_dat = []
    set_meta = {'num graphs': 0}

    # Set file names for set output
    if proc_id != '':
        savefile_set = f'{save_path}/{proc_id}_{targ_type}dat.pkl' # save_path is ../Data/{dataset}_procs_temp/
        metafile_set = f'{save_path}/{proc_id}_{targ_type}datmeta.json'
    else:
        savefile_set = save_path # save_path is ../Data/{dataset}_allobs.pkl
   
    # Loop through file set
    proc_name = f"[Proccess {proc_id}]" if proc_id != '' else ''
    print(f'\t{proc_name} Begining iteration through {len(alltrees_set)} trees. Will save to {savefile_set}', flush=True)
        
    # Combine trees and corresponding targets into data object  # data_z.py 205 - 295
    start = time.time()
    for rstar_id in list(alltrees_set.keys()): # Loop through tree keys becasue tree dict wont have any keys not in obs dict
        tree = alltrees_set[str(rstar_id)]
        # Create x tensor
        data = np.array(tree[featnames], dtype=float)
        X = torch.tensor(data, dtype=torch.float) 
        # Create y tensor 
        targs = allobs[rstar_id]
        y = torch.tensor(targs, dtype=torch.float) 
        # Create edge_index tensor (one tuple for every prog in the tree (all haloes except last))
        edge_index, edge_attr = make_edges(tree)
        # Combine into graph and add graph dat list
        graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
        set_dat.append(graph)  
        set_meta['num graphs'] += 1 
        print(f"\t{proc_name} Done with tree {set_meta['num graphs']}. Tree length {len(tree)}. Tot time {time.time()-start} s", flush=True)

    set_meta['total time'] = np.round(time.time() - start, 4)
    print(f"\t{proc_name} Done with all trees. Took {np.round(time.time() - start, 4)} s to save {set_meta['num graphs']} total graphs", flush=True)
    
    if proc_id != '': # If this is a thread, save set results. If not, just return them. 
        
        print(f'\t{proc_name} Saving graphs in {savefile_set} and saving processing meta to {metafile_set}', flush=True) 
        pickle.dump(set_dat, open(savefile_set, 'wb'))
        pickle.dump(set_meta, open(metafile_set, 'wb'))

        return [proc_id, set_meta]
    
    else:

        return alltrees_set, set_meta

def make_edges(tree):

    progs = []; descs = []
    for i in range(len(tree)):
        halo = tree.iloc[i]
        desc_id = halo['newdesc_id'] # try: desc_id = halo['newdesc_id']  except KeyError: desc_id = halo['olddesc_id'] # forgot to rename desc_id to newdesc_id in prep_trees, and then acidentally did keep the rename to olddesc_id
        if (desc_id != '-1'): 
            desc_pos = np.where(tree['id(1)']==desc_id)[0][0] 
            progs.append(i)
            descs.append(desc_pos) 
    edge_index = torch.tensor([progs,descs], dtype=torch.long) 
    edges = np.full(len(progs),np.NaN)
    edge_attr = torch.tensor(edges, dtype=torch.float)

    return edge_index, edge_attr

def response(result):
    result.append(result)

# def make_edges(tree):

#     progs = []; descs = []
#     for i in range(len(tree)):
#         halo = tree.iloc[i]
#         try: desc_id = halo['newdesc_id'] 
#         except KeyError: desc_id = halo['olddesc_id'] # forgot to rename desc_id to newdesc_id in prep_trees, and then acidentally did keep the rename to olddesc_id
#         if (desc_id != '-1'): 
#             desc_pos = np.where(tree['id(1)']==desc_id)[0][0] 
#             progs.append(i)
#             descs.append(desc_pos) 

#     return progs, descs

 # Collect other targets for the same galaxies as in the photometry
def collect_othertargs(obscat_path, alltrees, volname, save_path, reslim=100):
    '''
    THIS FUNCTION NEEDS TO GO AWAY: SHOULD BE DOING THE SAME THING AS RUNNING COLLECT_TARGS WITH TO_STORE DICT, EXCEPT FOR STORING AS {RSTAR_ID, list}, NOT JUST DF WITH RSTAR_ID COL
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

    # Add photometry for all central subhalos to dict [analogous to prep_phot]
    all_targs = pd.DataFrame(columns=['rstar_id']+list(to_store.keys()))
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

    # Collect just the galaxies that are in the stored trees [analogous to make_graphs]
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

# # Change dtypes MARCH 13
#  def change_dtypes(halos, featnames):
#      row = halos.iloc[0]
#      castto = []
#      for featname in featnames:
#          # dtype = "int64" if not isinstance(row[featname], str) else "float64" if '.' in row[featname] else "int64"
#          # dtype = type(row[featname]) if not isinstance(row[featname], str) else "float64" if '.' in row[featname] else "int64"
#          featvals = halos[featname]
#          containsfloatstr = any('.' in val for val in featvals)
#          containsnonstr = any(not isinstance(val, str) for val in featvals)
#          # if containsnonstr:
#          #     t = [type(val) for val in featvals if not isinstance(val, str)][0] # Lets see if i need this (like, are there any times when a row just has one or two non strings?)
#          dtype = type(featvals[0]) if containsnonstr else "float64" if containsfloatstr else "int64"
#          castto.append([featname, dtype])
#      halos = halos.astype(dict(castto))

#      return halos 


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
def scale(halos, featnames, lognames=['Mvir(10)', 'Mvir_all', 'M200b', 'M200c', 'M2500c', 'M_pe_Behroozi', 'M_pe_Diemer'], minmaxnames=['Jx(23)', 'Jy(24)', 'Jz(25)']):

    # Define scaling funcs
    def logt(x):
        return np.log10(x+1)
    def maxscale(x):
        return x/max(x)

    # Scale some features with logt # WHY WAS CHRISITIAN LOG SCALING NUM_PROG??
    tolog = [name for name in featnames if name in lognames]
    for name in tolog:
        halos[name] = logt(halos[name])

    # Scale some features with minmax 
    tominmax = [name for name in featnames if name in minmaxnames]
    for name in tominmax:
        halos[name] = maxscale(halos[name])

    return halos


def downsize_tree3(tree, num, add_gain=False, debug=False):
    '''
    Attempt to replicate Christian's method
    WTF why is this so much faster its like even way faster than Christians??? Cuts the same num halos as Christians (both way more than mine)
    NOTE: For easier to read version (which *should* be exactly the same, see downsize_tree3_test in test_funclib)
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
        descid = atree[:,descidcol][np.where(mid==atree[:,idcol])][0] # look at this halo's (descdendent's) descendent 
        while descid not in mergers[:,idcol] and descid!='-1': # if this halos (descdendent's) desc is not a merger or final, find the next halo who's desc is merger or final and assign it to be this halos be this halos new desc
            proid = atree[:,idcol][np.where(descid==atree[:,idcol])][0] # WAIT ISINT THIS THE SAME AS PROID = DESCID?? get the id of the prog of this halo's descendent (want to keep the progs of mergers - could this get moved outside while loop and just get it if this halo's desc is a merger?)   # added [0] 
            descid = atree[:,descidcol][np.where(descid==atree[:,idcol])][0] # look at this halo's descendent's descendent   # added [0] # desc id of halos desc  ##new descendant id where current descendant id = halo id
            k += 1
        if k>1 and proid != finalid: # if original desc of this merger was not a merger but its new desc is (as opposed to new desc being final (this is same as descid != -1, right?))
            des.append(proid); drc += 1
            halmix.append(atree[np.where(proid==atree[:,idcol])][0]); hrc +=1
            if debug:
                print(f"  Its original desc is not a merger or final, and this new desc is merger, not final")
                print(f"     Adding id of its new desc ('proid' {proid}) to des [des row {drc}]") # Add halo's new desc id at corresponding des index
                print(f"     Adding its new desc (halo at 'proid') to halmix [halmix row {hrc}]") # Add new descendent to halmix
        if descid!='-1': # if desc of this mergers new desc is not final (is a merger)
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

def downsize_tree4(tree, num, min_gainPct = 10, debug=False):
    '''
    Same as downsize_tree3 (really copied from downsize_tree3_test), except also retaining nodes if there is significant mass change
     - Add check for high mass gain to while loop
     - But also, don't exit while loop if desc is only high mass gain. 
       Still add it to newdescs at the index corresponding to the mid
       And then, like when premerger, still need to add it to out_halos and its own desc to newdecs
       But, 
    '''

    nprogcol = tree.columns.get_loc('num_prog(4)')
    didcol = tree.columns.get_loc('desc_id(3)')
    idcol = tree.columns.get_loc('id(1)')
    masscol = tree.columns.get_loc('Mvir(10)')

    atree= np.array(tree) # halwgal[n]
    roots = atree[atree[:,nprogcol]==0] 
    rids = roots[:,idcol]
    mergers = atree[atree[:,nprogcol]>1]
    mids = mergers[:,idcol]
    final = atree[atree[:,didcol]=='-1'][0] # [0] because atree[row] will give list inside a list
    finalid = final[idcol] 

    out_halos = []
    newdescs = []

    hrc = -1; drc = -1
    if debug: print(f"\n")
    if finalid not in mergers[:,idcol]: # IF FIRST HALO (FINAL HALO) IS NOT A MERGER (ONLY 1 DESC) IT ALSO WONT BE A ROOT SO IT WILL NEVER GET ADDED... 
        out_halos.append(list(final)); hrc += 1
        newdescs.append(final[didcol]); drc += 1
        if debug: print(f"Final halo {finalid} is not a merger, so adding it here.\n   Adding it to halmix [halmix row {hrc}]\n   Adding its desc id {final[didcol]} to des [des row {drc}]")
    if debug: print(f"Looping through roots and mergers")
    somemasskept = False

    for q, mid in enumerate(mids):

        if debug: print(f"Starting from halo of interest merger {q} (id {mid})")
        hid = mid # id of halo of interest
        end_of_chain = False
        while not end_of_chain:

            # Find check halo's original desc - if good, it will be either "end_of_chain" (merger, premerger, or final) or high mass gain
            orig_descid = atree[:,didcol][np.where(hid==atree[:,idcol])][0] # look at thiss halo's descendent
            descid = orig_descid
            descdescid = atree[atree[:,idcol]==descid][0][didcol] #if not desc_is_final else 'None'
            desc_is_final = descdescid =='-1' # check if candidate is a final
            desc_is_merger = descid in mids # Check if candidate is a merger
            desc_is_premerger = descdescid in mids # Check if candidate is a premerger (its desc is a merger)
            halo_mass = atree[:,masscol][np.where(hid==atree[:,idcol])][0] # this halo's mass
            gain_pct = 10**(halo_mass - atree[:,masscol][np.where(descid==atree[:,idcol])][0]) 
            desc_is_highgain = gain_pct > min_gainPct
            if debug: 
                if hid!=mid: print(f"New halo of interest {hid}")
                print(f"  Orig desc {orig_descid} is {'final' if descid==orig_descid and desc_is_final else 'merger' if desc_is_merger else 'premerger' if descid==orig_descid and desc_is_premerger else 'high mass change' if desc_is_highgain else 'not keepable'}")
            
            # Find new desc - it will be either "end_of_chain" (merger, premerger, or final) or high mass gain
            while not (desc_is_final or desc_is_merger or desc_is_premerger or desc_is_highgain): 
                halo_mass = atree[:,masscol][np.where(descid==atree[:,idcol])][0] # this halo (previous desc) mass
                descid = atree[:,didcol][np.where(descid==atree[:,idcol])][0] # step candidate descendent
                descdescid = atree[atree[:,idcol]==descid][0][didcol] #if not desc_is_final else 'None'
                desc_is_final = descdescid =='-1' # check if candidate is a final
                desc_is_merger = descid in mids # Check if candidate is a merger
                desc_is_premerger = descdescid in mids # Check if candidate is a premerger (its desc is a merger)
                gain_pct = 10**(halo_mass - atree[:,masscol][np.where(descid==atree[:,idcol])][0]) 
                desc_is_highgain = gain_pct > min_gainPct
            if desc_is_highgain: somemasskept = True

            # Add the merger to out_halos and its desc to newdescs
            if debug: 
                print(f"  Adding {'original merger' if hid==mid else 'halo of interest'} {hid} to halmix [halmix row {hrc}]") 
                if descid!=orig_descid: print(f"  Found new desc {descid} which is {'final' if desc_is_final else 'merger' if desc_is_merger else 'premerger' if desc_is_premerger else 'high mass change' if desc_is_highgain else 'error'}, adding it to newdescs [row {drc}]")
                else: print(f"  Adding original desc {descid} to newdescs [row {drc}]")
            out_halos.append(list(atree[np.where(hid==atree[:,idcol])][0])); hrc +=1 # add this halo of interest to out_halos
            newdescs.append(descid); drc+=1 # if descid=='-1' and '-1' in newdescs: raise ValueError(f"Halo {mid} has descid {descid}, but there is already a final halo added")
            unique_ids, counts = np.unique(np.array(out_halos)[:,idcol], return_counts=True)
            if np.any(counts > 1): raise ValueError(f"After adding merger {q} (id {mid}), tree_out has duplicate ids: {unique_ids[counts>1]}")

            # Add descendent to out_halos if it is a premerger and not also merger or final
            if desc_is_premerger and not (desc_is_merger or desc_is_final): 
                out_halos.append(list(atree[np.where(descid==atree[:,idcol])][0])); hrc +=1 # add this descendent to out_halos # if descid=='-1' and '-1' in newdescs: raise ValueError(f"Halo {mid} has descid {descid}, but there is already a final halo added")
                newdescs.append(descdescid); drc += 1 # add this descendent's descendent to newdescs
                if debug: print(f"  Since the {'orig' if descid==orig_descid else 'new'} desc is a premerger (and not merger), adding it to halmix [row {hrc}] and adding its own desc to newdescs [row {drc}]")

            # New desc is end_of_chain if it is good but NOT just for high mass gain, if not, set hid to descid and repeat
            end_of_chain = desc_is_final or desc_is_merger or desc_is_premerger 
            if not end_of_chain and desc_is_highgain: hid = descid


        # Checks
        if hrc != drc: raise ValueError(f"Error: hrc {hrc} != drc {drc}")
        if len(out_halos[-1]) != len(list(tree.columns)):
            raise ValueError(f"\n[haloset {num}] Error at out_halos row {hrc} (after proccessing merger {q}) has length {len(out_halos[-1])} not n feats {len(list(tree.columns))} \n\thalmix[-1] {out_halos[-1]}")

    c = 0     
    for q, rid in enumerate(rids):

        if debug: print(f"Starting from halo of interest root {q} (id {rid})")
        hid = rid # id of halo of interest
        end_of_chain = False
        while not end_of_chain:

            # Find check halo's original desc - if good, it will be either "end_of_chain" (merger, premerger, or final) or high mass gain
            halo_mass = atree[:,masscol][np.where(hid==atree[:,idcol])][0] # this halo's mass
            orig_descid = atree[:,didcol][np.where(hid==atree[:,idcol])][0] # look at thiss halo's descendent
            descid = orig_descid
            descdescid = atree[atree[:,idcol]==descid][0][didcol] #if not desc_is_final else 'None'
            desc_is_final = descdescid =='-1' # check if candidate is a final
            desc_is_merger = descid in mids # Check if candidate is a merger
            desc_is_premerger = descdescid in mids # Check if candidate is a premerger (its desc is a merger)
            gain_pct = 10**(halo_mass - atree[:,masscol][np.where(descid==atree[:,idcol])][0]) 
            desc_is_highgain = gain_pct > min_gainPct
            if debug: 
                if hid!=mid: print(f"New halo of interest {hid}")
                print(f"  Orig desc {orig_descid} is {'final' if descid==orig_descid and desc_is_final else 'merger' if desc_is_merger else 'premerger' if descid==orig_descid and desc_is_premerger else 'high mass change' if desc_is_highgain else 'not keepable'}")
            
            # Find new desc - it will be either "end_of_chain" (merger, premerger, or final) or high mass gain
            while not (desc_is_final or desc_is_merger or desc_is_premerger or desc_is_highgain): 
                descid = atree[:,didcol][np.where(descid==atree[:,idcol])][0] # step candidate descendent
                descdescid = atree[atree[:,idcol]==descid][0][didcol] #if not desc_is_final else 'None'
                desc_is_final = descdescid =='-1' # check if candidate is a final
                desc_is_merger = descid in mids # Check if candidate is a merger
                desc_is_premerger = descdescid in mids # Check if candidate is a premerger (its desc is a merger)
                gain_pct = 10**(halo_mass - atree[:,masscol][np.where(descid==atree[:,idcol])][0]) 
                desc_is_highgain = gain_pct > min_gainPct
            if debug and descid!=orig_descid: print(f"  Found new desc {descid} which is {'final' if desc_is_final else 'merger' if desc_is_merger else 'premerger' if desc_is_premerger else 'high mass change' if desc_is_highgain else 'error'}, adding it to newdescs")
            if desc_is_highgain: somemasskept = True

            # Add the root to out_halos and its desc to newdescs
            if debug: 
                print(f"  Adding {'original merger' if hid==rid else 'halo of interest'} {hid} to halmix [halmix row {hrc}]") 
                if descid!=orig_descid: print(f"  Found new desc {descid} which is {'final' if desc_is_final else 'merger' if desc_is_merger else 'premerger' if desc_is_premerger else 'high mass change' if desc_is_highgain else 'error'}, adding it to newdescs [row {drc}]")
                else: print(f"  Adding original desc {descid} to newdescs [row {drc}]")          
            out_halos.append(list(atree[np.where(hid==atree[:,idcol])][0])); hrc +=1 # add this halo of interest to out_halos
            newdescs.append(descid); drc+=1 # if descid=='-1' and '-1' in newdescs: raise ValueError(f"Halo {mid} has descid {descid}, but there is already a final halo added")
            unique_ids, counts = np.unique(np.array(out_halos)[:,idcol], return_counts=True)
            if np.any(counts > 1): raise ValueError(f"After adding root {q} (id {rid}), tree_out has duplicate ids: {unique_ids[counts>1]}")

            # Add descendent to out_halos if it is a premerger and not also merger or final
            if desc_is_premerger and not (desc_is_merger or desc_is_final): 
                out_halos.append(list(atree[np.where(descid==atree[:,idcol])][0])); hrc +=1 # add this descendent to out_halos # if descid=='-1' and '-1' in newdescs: raise ValueError(f"Halo {mid} has descid {descid}, but there is already a final halo added")
                newdescs.append(descdescid); drc += 1 # add this descendent's descendent to newdescs
                if debug: print(f"  Since the {'orig' if descid==orig_descid else 'new'} desc is a premerger (and not merger), adding it to halmix [row {hrc}] and adding its own desc to newdescs [row {drc}]")

            # New desc is end_of_chain if it is good but NOT just for high mass gain, if not, set hid to descid and repeat
            end_of_chain = desc_is_final or desc_is_merger or desc_is_premerger 
            if not end_of_chain and desc_is_highgain: hid = descid
        
        # Checks
        if hrc != drc: raise ValueError(f"Error: hrc {hrc} != drc {drc}")
        if len(out_halos[-1]) != len(list(tree.columns)):
            raise ValueError(f"\n[haloset {num}] Error: row at at out_halos row {hrc} (after proccessing root {c}) has length {len(out_halos[-1])} not n feats {len(list(tree.columns))} \n\thalmix[-1] {out_halos[-1]}")
        c += 1

    tree_out = pd.DataFrame(out_halos, columns=tree.columns)
    tree_out['newdesc_id'] = newdescs

    check_tree(tree_out, num=num)

    return tree_out, somemasskept


def downsize_tree5(intree, num, debug=False):
    '''
    Bottom-up downsize. Greatly increases efficiency. 
    '''

    # Get progenitor id col
    t0 = time.time()
    tree = intree.copy()
    tree = add_prog_ids(tree)


    ## Add jump ids 
    #final = tree[tree['desc_id(3)']=='-1']
    #jump_ids = {} #tree['jump_id'] = ['NaN' for i in range(len(tree))]
    #tree = add_jump_ids_rec(tree, start_halo=final, jump_ids=jump_ids, debug=debug)

    ## Build tree
    # tree_out = build_out_tree(tree)

    # Make tree array for easier indexing
    idcol = tree.columns.get_loc('id(1)')
    didcol = tree.columns.get_loc('desc_id(3)')
    pidcol = tree.columns.get_loc('prog_ids')
    mcol = tree.columns.get_loc('Mvir(10)')
    atree = np.array(tree) #tree.values.tolist()

    # Build tree
    t1 = time.time()
    out_halos = []
    new_descids = []
    final = atree[atree[:,didcol]=='-1']
    out_halos, new_descids = build_out_tree_rec(atree, final, out_halos, new_descids, idcol, didcol, pidcol, mcol, debug=debug)

    # Create DF
    t2 = time.time()
    tree_out = pd.DataFrame(out_halos, columns=tree.columns) # use intree cols (dont include prog_ids and jump_ids)
    tree_out['newdesc_id'] = new_descids
    tree_out = tree_out.drop(columns='prog_ids')

    #check_tree(tree_out, num) # THIS TAKES TOO LONG TO DO FOR ALL TREES

    #print(f"  Downsize tree5 - adding prog id col took {t1-t0:.2f} sec, building tree took {t2-t1:.2f} sec, making df {t3-t2:.2f} sec, checking tree took {time.time()-t3:.2f} sec")

    return tree_out

def build_out_tree_rec(tree, start_halo, out_halos, new_descids, idcol, didcol, pidcol, mcol, debug=False):

    # Get id of start halo in the "chain" (for first call, this is final z0 halo)
    halo_id = start_halo[0][idcol]# need [0] because outputs is [list]
    if debug: print(f"Called build_tree_rec with starting halo {halo_id}")
   
    # Add the start halo to out_halos and its desc to new_descids
    desc_id = start_halo[0][didcol]
    if debug: print(f"    Adding halo {halo_id} to out and desc {desc_id} to new_descs")
    out_halos.append(start_halo[0])
    new_descids.append(desc_id)
    
    # Set the "jump_id" to the start halo [only doing this to make the keeping of high mass gain halos easier, since then jump_id will change]
    jump_id = halo_id 

    # Find the end of the chain (next halo with more than one prog or no progs)
    prog_ids = start_halo[0][pidcol]
    while len(prog_ids) == 1: # if not keeping high mass gain, jump_id never changes in this loop
        if debug: print(f"    Halo {halo_id} has one prog ({prog_ids[0]}), so will move up chain to find new prog unless {prog_ids[0]} is a root")
        halo_id = prog_ids[0] # only one
        halo = tree[tree[:,idcol]==halo_id] # ANOTHER X N IN TIME 
        prog_ids = halo[0][pidcol] 

    # Unless the end of the chain is the starting halo itself (started with a merger), add it to out_halos and its desc to new_descids
    if halo_id != jump_id:
        if debug: print(f"  Found new prog {halo_id} which is a {'root' if len(prog_ids)==0 else 'merger'}. Adding halo {halo_id} to out and its desc (jump_id) {jump_id} to new_descids")
        out_halos.append(halo[0])
        new_descids.append(jump_id)

    # If the end of the chain is a merger, call build_tree_rec again with each prog
    if len(prog_ids) > 1:
        if debug: print(f"  Halo {halo_id} is a merger with progs {prog_ids}")
        for prog_id in prog_ids:
            if debug: print(f" Calling build_tree_rec again with start halo = {prog_id}")
            prog_halo = tree[tree[:,idcol]==prog_id] # ANOTHER X N IN TIME 
            build_out_tree_rec(tree, prog_halo, out_halos, new_descids, idcol, didcol, pidcol, mcol, debug)

    # If the end of the chain is a root, this branch is finished
    if len(prog_ids) == 0: 
        if debug: print(f"  Halo {halo_id} is a root, so this branch is finished")

    return out_halos, new_descids


def downsize_tree6(intree, num, debug=False):
    '''
    Even faster bottom-up downsize!
    '''

    # Add progenitor id col
    tree = intree.copy()
    tree = add_prog_ids(tree)

    # Make dict of {prog:desc} and {desc:progs}
    did_dict = dict(zip(tree['id(1)'], tree['desc_id(3)']))
    pid_dict = dict(zip(tree['id(1)'], tree['prog_ids']))

    # Build lists of halo ids to keep and their new desc ids
    out_hids = []
    new_descids = []
    final_id = tree[tree['desc_id(3)']=='-1']['id(1)'][0]
    out_hids, new_descids = build_keep_lists_rec(did_dict, pid_dict, final_id, out_hids, new_descids, debug=debug)
    
    # Create DF - this method is faster than looping through out_hids and appending to a list
    tree_out = tree.copy()
    tree_out['id'] = tree_out['id(1)']
    tree_out = tree_out.query(f'id in {out_hids}')  # to use query need to first rename id(1) to id 
    tree_out.set_index('id', inplace=True)
    tree_out = tree_out.reindex(index = out_hids)
    tree_out.reset_index(inplace=True)
    tree_out['newdesc_id'] = new_descids 
    tree_out = tree_out.drop(columns='prog_ids')

    #check_tree(tree_out, num) 

    return tree_out


def downsize_tree7(intree, check_gain_with, num=np.nan, debug=False):
    '''
    Even faster bottom-up downsize, with ability to retain high mass gain halos
    '''

    # Add progenitor id col
    tree = intree.copy()
    tree = add_prog_ids(tree)

    # Make dict of {prog:desc} and {desc:progs}
    did_dict = dict(zip(tree['id(1)'], tree['desc_id(3)']))
    pid_dict = dict(zip(tree['id(1)'], tree['prog_ids']))
    m_dict = dict(zip(tree['id(1)'], tree['Mvir(10)']))

    # Build lists of halo ids to keep and their new desc ids
    out_hids = []
    new_descids = []
    final_id = tree[tree['desc_id(3)']=='-1']['id(1)'][0]
    out_hids, new_descids, somemasskept = build_keep_lists_rec_7(did_dict, pid_dict, m_dict, final_id, out_hids, new_descids, check_gain_with=check_gain_with, debug=debug)
    
    # Create DF - this method is faster than looping through out_hids and appending to a list
    tree_out = tree.copy()
    tree_out['id'] = tree_out['id(1)']
    tree_out = tree_out.query(f'id in {out_hids}')  # to use query need to first rename id(1) to id 
    tree_out.set_index('id', inplace=True)
    tree_out = tree_out.reindex(index = out_hids)
    tree_out.reset_index(inplace=True)
    tree_out['newdesc_id'] = new_descids 
    tree_out = tree_out.drop(columns='prog_ids')

    #check_tree(tree_out, num) 
    
    return tree_out, somemasskept


def build_keep_lists_rec_7(did_dict, pid_dict, m_dict, start_id, out_hids, new_descids, check_gain_with, debug=False, somemasskept=False):

    halo_id = start_id
    if debug: print(f"Called build_tree_rec with starting halo {halo_id}")
    min_pct_gain=10
   
    # Add the start halo to out_hids and its desc to new_descids
    desc_id = did_dict[halo_id]
    if debug: print(f"    Adding halo {halo_id} to out and desc {desc_id} to new_descs")
    out_hids.append(halo_id)
    new_descids.append(desc_id)
    
    # Find the end of the chain (next halo with more than one prog, or high mass gain)
    jump_id = halo_id  # Set the "jump_id" to the start halo
    prog_ids = pid_dict[halo_id]
    pct_gain = 10**(m_dict[halo_id] - m_dict[prog_ids[0]]) if check_gain_with == 'prog' and len(prog_ids) == 1 else 0 # if checking gain with jump, set to 0 since this first halo is the jump
    while len(prog_ids) == 1 and pct_gain < min_pct_gain: 
        if debug: print(f"    Halo {halo_id} has one prog ({prog_ids[0]}), and small mass gain wrt that prog (10**({m_dict[halo_id] if check_gain_with == 'prog' else m_dict[jump_id]} - {m_dict[prog_ids[0]] if check_gain_with == 'prog' else m_dict[halo_id]} = {pct_gain} < {min_pct_gain})), so will move up chain to find new prog (unless {prog_ids[0]} is a root)")
        halo_id = prog_ids[0] # only one
        prog_ids = pid_dict[halo_id]
        if len(prog_ids) == 1: 
            if check_gain_with == 'prog': pct_gain = 10**(m_dict[halo_id] - m_dict[prog_ids[0]]) 
            elif check_gain_with == 'jump': pct_gain = 10**(m_dict[jump_id] - m_dict[halo_id]) # switch subtract order since now comparing to later halo (jump)
            else: raise ValueError(f"check_gain_with must be 'prog' or 'jump' not {check_gain_with}")
            
    # Unless the end of the chain is the starting halo itself (started with a merger or high mass gainer), add it to out_halos and its desc to new_descids
    if halo_id != jump_id:
        if pct_gain > min_pct_gain: somemasskept = True
        if debug: 
            if len(prog_ids)!=0: s = f"10**({m_dict[halo_id] if check_gain_with == 'prog' else m_dict[jump_id]} - {m_dict[prog_ids[0]] if check_gain_with == 'prog' else m_dict[halo_id]}) = {pct_gain} > {min_pct_gain})"
            print(f"  Found new prog {halo_id} which is a {'root' if len(prog_ids)==0 else 'merger' if len(prog_ids)>1 else f'high gain ({s})'}. Adding halo {halo_id} to out and its desc (jump_id) {jump_id} to new_descids")
        out_hids.append(halo_id)
        new_descids.append(jump_id)

    # If the end of the chain is a merger or a high mass gain, call build_tree_rec again with each prog
    if len(prog_ids) > 1 or pct_gain > min_pct_gain:
        if debug: print(f"  Halo {halo_id} is a {'merger with progs' if len(prog_ids) > 1 else 'high gain with prog'} {prog_ids}")
        for prog_id in prog_ids:
            if debug: print(f" Calling build_tree_rec again with start halo = {prog_id}, some mass kept = {somemasskept}")
            build_keep_lists_rec_7(did_dict, pid_dict, m_dict, prog_id, out_hids, new_descids, check_gain_with, debug, somemasskept)

    # If the end of the chain is a root, this branch is finished
    if len(prog_ids) == 0: 
        if debug: print(f"  Halo {halo_id} is a root, so this branch is finished")

    return out_hids, new_descids, somemasskept

def build_keep_lists_rec(did_dict, pid_dict, start_id, out_hids, new_descids, debug=False):

    hid = start_id
    if debug: print(f"Called build_tree_rec with starting halo {hid}")
   
    if did_dict[hid] == '-1': 
        if debug: print(f"  Start halo {hid} is final, so adding {hid} to out and -1 to new_descids")
        out_hids.append(hid)
        new_descids.append('-1')
    
    jump_id = hid 
    prog_ids = pid_dict[hid]

    while len(prog_ids) == 1: # if not keeping high mass gain, jump_id never changes in this loop
        if debug: print(f"    Halo {hid} has one prog ({prog_ids[0]}), so will move up chain to find new prog unless {prog_ids[0]} is a root")
        hid = prog_ids[0] # only one
        prog_ids = pid_dict[hid]

    if hid != jump_id:
        if debug: print(f"  Found new prog {hid} which is a {'root' if len(prog_ids)==0 else 'merger'}. Adding halo {hid} to out and its desc (jump_id) {jump_id} to new_descids")
        out_hids.append(hid)
        new_descids.append(jump_id)

    if len(prog_ids) > 1:
        if debug: print(f"  Halo {hid} is a merger with progs {prog_ids}")
        for prog_id in prog_ids:
            if debug: print(f"    Adding prog {prog_id} to out and id {hid} to new_descs.\n    Calling build_tree_rec again with start halo = {prog_id}")
            out_hids.append(prog_id)
            new_descids.append(hid)
            build_keep_lists_rec(did_dict, pid_dict, prog_id, out_hids, new_descids, debug)

    if len(prog_ids) == 0: 
        if debug: print(f"  Halo {hid} is a root, so this branch is finished")

    return out_hids, new_descids

  
def add_prog_ids(tree):

    ids = tree['id(1)']
    desc_ids = tree['desc_id(3)']
    halo_idx = dict(zip(np.array(ids), np.linspace(0, len(ids)-1, len(ids), dtype=int)))
    prog_ids = [[] for i in range(len(tree))]
    for i in range(len(ids)):
        descid = desc_ids[i]
        if descid != '-1':
            idx_of_desc = halo_idx[descid] # I think the key is that this is significantly faster than idx_of_desc = tree[tree['id(1)']==descid] ??
            prog_ids[idx_of_desc].append(ids[i])

    tree_out = tree.copy()
    tree_out['prog_ids'] = prog_ids

    return tree_out

def downsize_tree8(tree, num, min_pct_gain=10, debug=False):
    '''
    Lehman's method
    NOTE: Doenst keep premergers
    '''

    # Get new progenitors of each halo
    desc_dict = {int(h): int(d) for h, d in zip(tree['id(1)'], tree['desc_id(3)']) if d != '-1'}
    nprog_dict = defaultdict(int)
    for d in desc_dict.values():
        nprog_dict[d] += 1
    desc_dict = simplify_tree_iterative(desc_dict, nprog_dict, debug)


    # Create tree
    tree_out = tree.copy()
    out_hids = [tree[tree['desc_id(3)']=='-1']['id(1)'][0]] + [str(k) for k in desc_dict.keys()] # add back in prog of final, which is not in desc_dict
    new_descids = ['-1'] + [str(v) for v in desc_dict.values()] # add back in final, which is not in desc_dict
    tree_out['id'] = tree_out['id(1)']
    tree_out = tree_out.query(f'id in {out_hids}')  # to use query need to first rename id(1) to id 
    tree_out.set_index('id', inplace=True)
    tree_out = tree_out.reindex(index = out_hids)
    tree_out.reset_index(inplace=True)
    tree_out['newdesc_id'] = new_descids 
    
    return tree_out

def simplify_tree_iterative(desc_dict, nprog_dict, debug=False):
    """Replace nodes that only have one descendant and one progenitor
    with an edge, using an iterative algorithm.
    """

    i = 0
    hids = list(desc_dict.keys())
    while i < len(hids):
        hid = hids[i]
        if hid not in desc_dict:
            if debug: print(f"hid {hid} doesnt have a desc (is final or has been deleted from desc_dict), so skipping")
            i += 1
            continue
        d = desc_dict[hid]
        if nprog_dict.get(d, 0) == 1 and d in desc_dict:
            desc_dict[hid] = desc_dict[d]
            if debug: print(f"hid {hid} has desc {d} which has one prog and a desc, so setting the desc of {hid} to the desc of {d} ({desc_dict[d]}) and deleting {d} from desc and nprog")
            del desc_dict[d], nprog_dict[d]
        else:
            i += 1

    return desc_dict


# Downsize function
def simplify_tree_iterative_old(desc_dict, nprog_dict, debug):
    """Replace nodes that only have one descendant and one progenitor
    with an edge, using an iterative algorithm.
    """

    dirty = True
    while dirty:
        dirty = False
        for hid in set(desc_dict):
            if hid not in desc_dict:
                if debug: print(f"hid {hid} doesnt have a desc (is final or has been deleted from desc_dict), so skipping")
                continue
            d = desc_dict[hid]
            if nprog_dict.get(d, 0) == 1 and d in desc_dict:
                if debug: print(f"hid {hid} has desc {d} which has one prog and a desc, so setting the desc of {hid} to the desc of {d} ({desc_dict[d]}) and deleting {d} from desc and nprog")
                desc_dict[hid] = desc_dict[d]
                del desc_dict[d], nprog_dict[d]
                dirty = True
    
    return desc_dict


# Function for simple check of tree outputs
def check_tree(tree, num, print_good=False):

    for i in range(len(tree)):
        descid = tree.iloc[i]['newdesc_id']
        if descid not in list(tree['id(1)']) and descid != '-1': raise ValueError(f"[Tree {num}] Halo {tree.iloc[i]['id(1)']} has desc {descid}, which is not in the tree")
    try: 
        np.array(tree, dtype=float)
    except: 
        print(f"[Tree {num}] Cant set tree to array")
        for i in range(len(tree)):
            try: np.array(tree.iloc[i], dtype=float)
            except: raise ValueError(f'[Tree {num}] Error is at row {i}\n\t {list(tree.iloc[i:i+1])}')
    for i in range(len(tree)):
        desc_id = tree.iloc[i]['newdesc_id']
        if type(desc_id) != str: raise ValueError(f"[Tree {num}] Error at row {i}: tree_out.iloc[{i}]['newdesc_id'] ({tree.iloc[i]['newdesc_id']}) is not a string")
        if not (desc_id in list(tree['id(1)']) or desc_id == '-1'):
            raise ValueError(f"[Tree {num}] Error at row {i}: tree_out.iloc[{i}]['newdesc_id'] ({tree.iloc[i]['newdesc_id']}) not in list(tree_out['id(1)'])\n\tlist(tree_out['id(1)']):\n\t\t{list(tree['id(1)'])}\n\tlist(tree_out['newdesc_id']):\n\t\t{list(tree['newdesc_id'])}")
    unique_ids, counts = np.unique(tree['id(1)'], return_counts=True) # np.unique(tree_out['id(1)'], return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"[Tree {num}] Tree has duplicate ids")

    roots = tree[tree['num_prog(4)']==0]
    for i in range(len(roots)):
        root = roots.iloc[i]
        desc_id = root['newdesc_id']
        seen = 0
        while desc_id != '-1' and seen < len(tree):
            desc_id = tree[tree['id(1)']==desc_id]['newdesc_id'].values[0]
            seen += 1
        if seen == len(tree):
            raise ValueError(f"[Tree ] Root {i} (hid {root['id(1)']}) has a descendent chain that never reaches final")
            
    data = np.array(tree, dtype=float)
    X = torch.tensor(data, dtype=torch.float) 
    y = torch.tensor(np.nan, dtype=torch.float) 
    edge_index, edge_attr = make_edges(tree)
    graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if X.shape[0]-1 != len(edge_attr): 
        raise Exception(f"[Tree {num}] Number of edges ({len(edge_attr)}) is not equal to one less than the number of nodes in graph ({X.shape[0]})")
    try:
        G = tg.utils.to_networkx(graph)
    except:
        raise Exception(f"[Tree {num}] Tree cannot be made into a graph") 
    # if not nx.is_tree(G):
    #     raise TypeError('Graph is not a tree')
    if len(G) == 0:
        raise Exception("G has no nodes")
    if not G.is_directed():
        raise Exception("G is not directed")
    if not nx.is_weakly_connected(G):
        raise Exception("G is not connected")

    if print_good: print('Downsized tree passed all tests')


########################
# Sanity check functions
########################

def combine_all_data(dataset):
    '''
    For DS2, for which I made allprops analogous to allobs
    Output is same as props graphs, except instead of [tree, props] just [mhaloz0, props, phot], and stored in DF
    '''

    alltrees = pickle.load(open(osp.expanduser(dataset.replace('.pkl','_alltrees.pkl')), 'rb'))
    allprops = pickle.load(open(osp.expanduser(dataset.replace('.pkl','_allprops.pkl')), 'rb'))
    allphot = pickle.load(open(osp.expanduser(dataset.replace('.pkl','_allphot.pkl')), 'rb'))

    info = pd.DataFrame(columns=['rstar_id', 'mhaloz0', 'tree_len' ,'props', 'phot'])
    for id in alltrees.keys():
        # Info from tree
        mhaloz0 = alltrees[id].loc[0]['Mvir(10)']
        tree_len = len(alltrees[id])
        # Info from SF
        props = allprops[id] # mstar = allprops[id][mstar_idx]
        phot = allphot[id] # Umag = allphot[id][0]; Vmag = allphot[id][2]
        # Combine
        info.loc[len(info)] = {'rstar_id': id, 'mhaloz0': mhaloz0, 'tree_len': tree_len ,'props': props, 'phot': phot} #'umag': Umag, 'vmag': Vmag}

    pickle.dump(info, open(dataset.replace('.pkl','_all_data_combined.pkl'), 'wb'))

    return info


def prep_mstar(obscat_path, volname, save_path):
    '''
    For DS1, for which I didnt make allprops analogous to allobs
    '''
    
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
    '''
    For DS1, for which I didnt make allprops analogous to allobs
    '''

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
    print(f"Dataset {dataset}")
    print(f"   Volume: {data_meta[0]['set meta']['Volume']}")
    print(f"   Cuts: sizelim = {tree_meta['sizelim']}, Mlim = {tree_meta['Mlim']}, reslim = {obs_meta['reslim']}")
    print(f"   {obs_meta['tot galaxies']} total galaxies in phot catalog, {obs_meta['tot galaxies saved']} saved")
    print(f"        {obs_meta['galaxies res cut']} too low res, {obs_meta['galaxies sat cut']} not central ")
    print(f"   {tree_meta['tot trees']} total trees in ctrees files, {tree_meta['tot trees saved']} saved")
    print(f"        {tree_meta['trees mass cut']} with log(M) < {tree_meta['Mlim']}, {tree_meta['trees sat cut']} not central, {tree_meta['trees no mergers cut']} without mergers, {tree_meta['trees size cut']} with num halos > {tree_meta['sizelim']}")

#############################
# Loading data for train/test 
#############################

def create_subset(data_params, return_set, save_extras=True): 
    '''
    Load data from pickle, transform, remove unwanted features and targets, and split into train and val or train and test
    Save two pickle files
        1. *_train.pkl: contains two objects. One is list of train graphs, the other is list of val graphs.
        2. *_test.pkl: contains two objects. One is list of test graphs, the other is list of additional z0 x data for these test graphs
    '''

    # Use dataset meta file to see names of all feats and targs in dataset
    data_path = data_params['data_path'] # e.g. ../Data/vol100/
    datafile = data_params['data_file'] # e.g. DS2_phot_final.pkl
    dataset = datafile[0:datafile.find('_')] # e.g. DS2
    targ_type = datafile[datafile.find('_')+1:datafile.find('_fin')] # e.g. phot
    data_meta_file = osp.expanduser(osp.join(data_path, f'{dataset}_meta.json'))
    data_meta = json.load(open(data_meta_file, 'rb'))
    targs_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'{targ_type} meta')[0][0]
    all_labelnames = data_meta[targs_idx][f'{targ_type} meta']['label_names'] 
    tree_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'tree meta')[0][0]
    all_featnames = data_meta[tree_idx]['tree meta']['featnames']

    # Load dataset subset parameters 
    use_targidxs = [all_labelnames.index(targ) for targ in data_params['use_targs']] # targets = data_params['targets']
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in data_params['use_feats']] #  # have use_feats be names not idxs! in case we want to train on fewer features without recreating dataset
    transfname = data_params['transf_name']
    splits = data_params['splits']

    # Load data from pickle
    print(f'Creating subset from {data_path}/{datafile}', flush=True)
    data = pickle.load(open(osp.expanduser(f'{data_path}/{datafile}'), 'rb'))
    # if targ_type == 'props':
    #     data = data[0] # for some reason getting saved as [data]

    # Split into trainval and test
    testidx_file = osp.expanduser(f'{data_path}/{dataset}_testidx{splits[2]}.npy')
    if not os.path.exists(testidx_file): 
        print(f'   Test indices for {splits[2]}% test set not yet saved for this dataset ({datafile}). Creating and saving in {testidx_file}', flush=True)
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
                
    # Fit transformers to all trainval x data and all test x data. Note: transformer is for x data, so use same one for props and phot sets. Note: Use all cols for transform 
    if transfname != "None":
        
        traintransformer_path = osp.expanduser(f'{data_path}/{dataset}_{transfname}_train.pkl')
        if not os.path.exists(traintransformer_path):
            print(f'   Fitting transformer to train/val data (will save in {traintransformer_path})', flush=True)
            ft_train = run_utils.fit_transformer(trainval_data, all_featnames, save_path=traintransformer_path, transfname=data_params['transf_name']) # maybe should do seperate for train/val too, but as long as not fitting on final test its probabaly ok
        else: 
            print(f'   Applying transformer to train/val data (loaded from {traintransformer_path})', flush=True)
            ft_train = pickle.load(open(traintransformer_path, 'rb')) 
        
        testtransformer_path = osp.expanduser(f'{data_path}/{dataset}_{transfname}_test.pkl')
        if not os.path.exists(testtransformer_path):
            print(f'   Fitting transformer to test data (will save in {testtransformer_path})', flush=True)
            ft_test = run_utils.fit_transformer(test_data, all_featnames, save_path=testtransformer_path, transfname=data_params['transf_name']) 
        else: 
            print(f'   Applying transformer to test data (loaded from {testtransformer_path})', flush=True)
            ft_test = pickle.load(open(testtransformer_path, 'rb')) 

    # Transform each graph, and remove unwanted features and targets
    print(f"   Selecting desired features {data_params['use_feats']} and targets {data_params['use_targs']}", flush=True)
    trainval_out = []
    for graph in trainval_data:
        x = graph.x
        if transfname != "None": 
            x = ft_train.transform(graph.x)
        x = torch.tensor(x[:, use_featidxs])
        y = graph.y[use_targidxs]
        trainval_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=y))
    test_out = []
    test_z0x_all = []
    for graph in test_data:
        x = graph.x
        test_z0x_all.append(x[0]) # add all feats from final halo
        if transfname != "None": 
            x = ft_test.transform(x)
        x = torch.tensor(x[:, use_featidxs])
        y = graph.y[use_targidxs]
        test_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=y))

    # Split trainval into train and val [MAYBE I SHOULD BE RANDOMIZING THIS]
    print(f"   Splitting train/val into train and val", flush=True)
    val_pct = splits[1]/(splits[0]+splits[1]) # pct of train from total percents [70, 10, 20]
    # train_out = trainval_out[:int(len(trainval_out)*(1-val_pct))]
    # val_out = trainval_out[int(len(trainval_out)*(val_pct)):]
    val_idxs = np.random.choice(len(trainval_out), int(len(trainval_out)*val_pct), replace=False) #trn_idxs = [i for i in range(len(trainval_X)) if i not in val_idxs]
    train_out, val_out = [], []
    for i in range(len(trainval_out)):
        if i in val_idxs:
            val_out.append(trainval_out[i])
        else:
            train_out.append(trainval_out[i])
    print(f"   Train set has {len(train_out)} graphs, Val set has {len(val_out)} graphs, Test set has {len(test_out)} graphs", flush=True)

    # Save (seperately for train and test to aviod any mistakes!)
    tag = f"{targ_type}_{use_featidxs}_{use_targidxs}_{transfname}_{splits}"
    train_path = osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_train.pkl') 
    test_path = osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_test.pkl') 
    print(f"   Saving train+val sub-dataset as {train_path}", flush=True)
    print(f"   Saving test sub-dataset as {test_path}", flush=True)
    pickle.dump((train_out, val_out), open(train_path, 'wb'))
    if save_extras:
        pickle.dump((test_out, test_z0x_all), open(test_path, 'wb')) # also save z0 feats for all test halos, for future exploration
    else: pickle.dump(test_out, open(test_path, 'wb')) 

    # If "test" run return train+val and test (NOTE: Christian uses 'test' to mean training the same model again completely from scratch on train+val, then testing on test)
    if return_set == "test":
        return trainval_out, test_out, test_z0x_all 
    if return_set == 'train':
        return train_out, val_out 
    else: raise ValueError('Must select either train or test set to return')
    
    
def get_subset_path(data_params, set):

    # Get subset parameters
    data_path = data_params['data_path'] # e.g. ../Data/vol100/
    datafile = data_params['data_file'] # e.g. DS2_phot_final.pkl
    dataset = datafile[0:datafile.find('_')] # e.g. DS2
    targ_type = datafile[datafile.find('_')+1:datafile.find('_fin')] # e.g. phot
    data_meta_file = osp.expanduser(osp.join(data_path, f'{dataset}_meta.json'))
    data_meta = json.load(open(data_meta_file, 'rb'))
    targs_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'{targ_type} meta')[0][0]
    all_labelnames = data_meta[targs_idx][f'{targ_type} meta']['label_names'] 
    tree_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'tree meta')[0][0]
    all_featnames = data_meta[tree_idx]['tree meta']['featnames']
    use_targidxs = [all_labelnames.index(targ) for targ in data_params['use_targs']] 
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in data_params['use_feats']] 
    transfname = data_params['transf_name']
    splits = data_params['splits']
    tag = f"{targ_type}_{use_featidxs}_{use_targidxs}_{transfname}_{splits}"

    # Return train or test path
    if set =='test':
        return osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_test.pkl')  
    else:
        return osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_train.pkl')  

# Helper function to read meta file regardless of dict positions (e.g. works for datasets with and without props meta)
def read_data_meta(metafile, targ_type):

    if targ_type == 'samProps': targ_type = 'targ'

    data_meta = json.load(open(metafile, 'rb'))
    tree_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'tree meta')[0][0]
    targs_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'{targ_type} meta')[0][0]
    graph_idx = np.where(np.array([list(data_meta[i].keys())[0] for i in range(len(data_meta))]) == f'final {targ_type} meta')[0][0]

    set_meta = data_meta[0]['set meta']
    tree_meta =  data_meta[tree_idx]['tree meta']
    targ_meta = data_meta[targs_idx][f'{targ_type} meta']
    graph_meta = data_meta[graph_idx][f'final {targ_type} meta']

    return set_meta, tree_meta, targ_meta, graph_meta


# debugging helper function
def load_like_prep(file, featnames = ['#scale(0)', 'desc_scale(2)', 'num_prog(4)', 'Mvir(10)']):

    path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L75n1820TNG_DM/postprocessing/trees/consistent-trees/' # 127 files
    # all_names = ['#scale(0)', 'id(1)', 'desc_scale(2)', 'desc_id(3)', 'num_prog(4)', 'pid(5)', 'upid(6)', 'desc_pid(7)', 'phantom(8)', 'sam_Mvir(9)', 'Mvir(10)', 'Rvir(11)', 'rs(12)', 'vrms(13)', 'mmp?(14)', 'scale_of_last_MM(15)', 'vmax(16)', 'x(17)', 'y(18)', 'z(19)', 'vx(20)', 'vy(21)', 'vz(22)', 'Jx(23)', 'Jy(24)', 'Jz(25)', 'Spin(26)', 'Breadth_first_ID(27)', 'Depth_first_ID(28)', 'Tree_root_ID(29)', 'Orig_halo_ID(30)', 'Snap_idx(31)', 'Next_coprogenitor_depthfirst_ID(32)', 'Last_progenitor_depthfirst_ID(33)', 'Last_mainleaf_depthfirst_ID(34)', 'Tidal_Force(35)', 'Tidal_ID(36)', 'Rs_Klypin', 'Mvir_all', 'M200b', 'M200c', 'M500c', 'M2500c', 'Xoff', 'Voff', 'Spin_Bullock', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]', 'A[z]', 'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', 'A[y](500c)', 'A[z](500c)', 'T/|U|', 'M_pe_Behroozi', 'M_pe_Diemer', 'Halfmass_Radius']
    raw = pd.read_table(f'{path}/{file}', header=0, skiprows=0, delimiter='\s+', usecols = np.linspace(0,30,31, dtype=int), dtype=str).drop(axis=0, index=np.arange(48))
    halos = raw[~raw.isna()['desc_id(3)']] # rows (nodes)
    halos = change_dtypes(halos, featnames) 
    halos = make_zcut(halos, zcut=('before', np.inf))
    halos = scale(halos, featnames)

    return halos


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
# class SimpleData(Dataset_notgeom):

#   def __init__(self, x, y):
#     self.x = x
#     self.y = y
#     self.len = len(x)
  
#   def __getitem__(self, index):
#     return self.x[index], self.y[index]
  
#   def __len__(self):
#     return self.len

def data_finalhalos(datapath, dataset, targ_type, use_feats, use_targs, savepath):

    # Load data, testidxs, and transforms
    if not 'sam' in targ_type:
        datafile = f"{datapath}/{dataset}_{targ_type}_final.pkl"
        ft_train = pickle.load(open(f"{datapath}/{dataset}_QuantileTransformer_train.pkl" , 'rb'))
        ft_test = pickle.load(open(f"{datapath}/{dataset}_QuantileTransformer_test.pkl" , 'rb'))
    else: 
        datafile = f"{datapath}/{dataset}.pkl" # for now, not transforming sam data 
    testidx_file = f'{datapath}/{dataset}_testidx20.npy' # probably unlikely I'll change split
    print(f"  Loading full dataset from {datafile}, corresponding test idxs file, and train and test transformers")
    data = pickle.load(open(datafile, 'rb')) # graphs

    # Get features and targs to use
    _, tree_meta, phot_meta, _ = read_data_meta(f"{datapath}/{dataset}_meta.json", targ_type)
    all_labelnames = phot_meta['label_names'] 
    all_featnames = tree_meta['featnames']
    use_targidxs = [all_labelnames.index(t) for t in all_labelnames if t in use_targs] 
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in use_feats]
    print(f"  Using feats {use_feats} (idxs {use_featidxs})")
    
    # # Transform and remove unwanted features and targets
    # X = []; Y = []
    # for graph in allgraphs:
    #     x = np.expand_dims(graph.x[0].detach().numpy(), axis=0)
    #     if not 'sam' in targ_type: x = ft_train.transform(x).squeeze() # for now, not transforming sam data 
    #     else: x = x.squeeze()
    #     X.append(x[use_featidxs].astype(np.float32))
    #     Y.append(graph.y[use_targidxs].detach().numpy().astype(np.float32))
    
    # # Split into trainval and test
    # print(f'  Loading test indices from {testidx_file}', flush=True)
    # testidx = np.array(np.load(testidx_file))
    # X_trainval, X_test = [], []
    # Y_trainval, Y_test = [], []
    # for i in range(len(X)):
    #     if i in testidx:
    #         X_test.append(X[i])
    #         Y_test.append(Y[i])
    #     else:
    #         X_trainval.append(X[i])
    #         Y_trainval.append(Y[i])

    # Split into trainval and test
    print(f'  Loading test indices from {testidx_file}', flush=True)
    testidx = np.array(np.load(testidx_file))
    test_data = []
    trainval_data = [] 
    for i, d in enumerate(data):
        if i in testidx:
            test_data.append(d)
        else:
            trainval_data.append(d)

    # Transform each graph, and remove unwanted features and targets
    print(f"   Selecting desired features {use_feats} and targets {use_targs}", flush=True)
    trainval_X, trainval_Y = [], []
    for graph in trainval_data:
        x = graph.x 
        if not 'sam' in targ_type: # for now, not transforming sam data 
            x = ft_train.transform(x)
        trainval_X.append(torch.tensor(x[:, use_featidxs].astype(np.float32))[0]) # [0] for final halo
        trainval_Y.append(graph.y[use_targidxs])
    test_X, test_Y = [], []
    for graph in test_data:
        x = graph.x
        if not 'sam' in targ_type: # for now, not transforming sam data 
            x = ft_test.transform(x)
        test_X.append(torch.tensor(x[:, use_featidxs].astype(np.float32))[0]) # [0] for final halo
        test_Y.append(graph.y[use_targidxs])
            
    # # Transform each graph, and remove unwanted features and targets
    # print(f"   Selecting desired features {use_feats} and targets {use_targs}", flush=True)
    # trainval_X, trainval_Y = [], []
    # for graph in trainval_data:
    #     x = graph.x
    #     if not 'sam' in targ_type: # for now, not transforming sam data 
    #         x = ft_train.transform(x)
    #     trainval_X.append(torch.tensor(x[:, use_featidxs]))
    #     trainval_Y.append(graph.y[use_targidxs])
    # test_X, test_Y = [], []
    # for graph in test_data:
    #     x = graph.x
    #     if not 'sam' in targ_type: # for now, not transforming sam data 
    #         x = ft_test.transform(x)
    #     trainval_X.append(torch.tensor(x[:, use_featidxs]))
    #     trainval_Y.append(graph.y[use_targidxs])

    # Split trainval into train and val 
    print(f"   Splitting train/val into train and val", flush=True)
    val_pct = 10/(10+70) 
    # train_X = trainval_X[:int(len(trainval_X)*(1-val_pct))]
    # train_Y = trainval_Y[:int(len(trainval_Y)*(1-val_pct))]
    # val_X = trainval_X[int(len(trainval_X)*(val_pct)):]
    # val_Y = trainval_Y[int(len(trainval_Y)*(val_pct)):]
    val_idxs = np.random.choice(len(trainval_X), int(len(trainval_X)*val_pct), replace=False) #trn_idxs = [i for i in range(len(trainval_X)) if i not in val_idxs]
    train_X, train_Y, val_X, val_Y = [], [], [], []
    for i in range(len(trainval_data)):
        if i in val_idxs:
            val_X.append(trainval_X[i])
            val_Y.append(trainval_Y[i])
        else:
            train_X.append(trainval_X[i])
            train_Y.append(trainval_Y[i])

    # # Save (seperately for train and test to aviod any mistakes!)
    # tag = f"{targ_type}_{use_featidxs}_{use_targidxs}_{transfname}_{splits}"
    # train_path = osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_train.pkl') 
    # test_path = osp.expanduser(f'{data_path}/{dataset}_subsets/{dataset}_{tag}_test.pkl') 
    # print(f"   Saving train+val sub-dataset as {train_path}", flush=True)
    # print(f"   Saving test sub-dataset as {test_path}", flush=True)
    # pickle.dump((train_out, val_out), open(train_path, 'wb'))
    # if save_extras:
    #     pickle.dump((test_out, test_z0x_all), open(test_path, 'wb')) # also save z0 feats for all test halos, for future exploration
    # else: pickle.dump(test_out, open(test_path, 'wb')) 

    traindata = SimpleData(train_X, train_Y)
    valdata = SimpleData(val_X, val_Y)
    testdata = SimpleData(test_X, test_Y)

    #pickle.dump((train_X, train_Y, val_X, val_Y,test_X, test_Y), open(savepath.replace('.pkl', '_seperate.pkl'), 'wb'))
    pickle.dump((traindata, valdata, testdata), open(f'{savepath}', 'wb')) 
  
    return 


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

def sampropsdata_finalhalos(props_data, use_featidxs, use_targidxs):

    X = []; Y = []
    for graph in props_data:
        X.append(graph.x[0].detach().numpy()[use_featidxs].astype(np.float32))
        Y.append(graph.y[use_targidxs].detach().numpy().astype(np.float32))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    traindata = SimpleData(X_train, Y_train)
    testdata = SimpleData(X_test, Y_test)
  
    return traindata, testdata


def get_yhats(preds, sample_method='large_sample', N=100):
    '''
    Get point estimates from predictions
        If predictions are single array of predicted means, return that array 
        If predictions are a tuple of arrays of predicted means and SDs (not log SDs - already exponentiated)
            If 'method' == 'single', return a single sample from the predicted distribution of each observation
            If 'method' == 'average', return average of a 1000 samples from the predicted distribution of each observation (which is sort of just a point estimate for the mean, as N->large)
            If 'method' == 'large_sample', return 1000 samples from the predicted distribution of each observation
    '''

    if len(preds) == 2: 
        muhats = preds[0]

        if sample_method == 'no_sample':
            yhats = muhats

        else:
            sigmahats = preds[1]
            # if sample_method == 'single_': 
            #     yhats = np.random.normal(muhats, sigmahats) 
            # else:
            y_samples = [np.random.normal(muhats, sigmahats) for i in range(N)]

            if sample_method == 'average_large_sample':
                yhats = np.mean(y_samples, axis=0)
            elif sample_method == 'large_sample':
                yhats = np.array(y_samples)

            else: raise ValueError(f'method must be no_sample, average_large_sample, or large_sample not {sample_method}')  

    else: 
        yhats = preds

    return yhats

def flatten_sampled_yhats(yhats, ys):
    '''
    If yhats represent N samples from each distribution, create flattened yhats and corresponding ys vectors
    '''

    N = yhats.shape[0] # number of samples per observation
    p = yhats.shape[1] # number of observations
    q = yhats.shape[2] # number of targets
    if p < N < q:
        raise ValueError('Check the shape of yhats. Should be (n samples per observation, n observations, n targets)')
    yhat_samples_flat = np.zeros((N*p,q))
    y_samples_flat = np.zeros((N*p,q))
    for i in range(len(yhats)):
        yhat_samples_flat[i*p:(i+1)*p] = yhats[i]
        y_samples_flat[i*p:(i+1)*p] = ys

    return np.squeeze(yhat_samples_flat), np.squeeze(y_samples_flat)

def sample_results(ys, preds, n_samples):

    if len(preds) != 2: 
        raise ValueError('Doesnt look like predictions are from a distribution-predicing model')
    muhats = preds[0]
    sigmahats = preds[1]
    rhos =[]; rmses = []; R2s = []; biases = []; scatters = []
    for n in range(n_samples):
        yhats = np.random.normal(muhats, sigmahats) # pt estimate from each predicted distribution for each target
        rmses.append(np.std(yhats - ys, axis=0)) # RMSE for each targ
        rhos.append(np.array([stats.pearsonr(ys[:,i], yhats[:,i]).statistic for i in range(ys.shape[1])])) # Pearson R for each targ
        R2s.append(np.array([skmetrics.r2_score(ys[:,i], yhats[:,i]) for i in range(ys.shape[1])])) # coefficient of determination
        biases.append(np.mean(ys-yhats, axis=0))  # mean of the residuals
        scatters.append(np.std(ys-yhats, axis=0)) # SD of the residuals

    # res = {'rho':np.mean(rhos, axis=0), # mean over all samples
    #        'rmse':np.mean(rmses, axis=0), # mean over all samples
    #        'R2':np.mean(R2s, axis=0), # mean over all samples
    #        'bias':np.mean(biases, axis=0)} # mean over all samples

    return rhos, rmses, R2s, biases, scatters