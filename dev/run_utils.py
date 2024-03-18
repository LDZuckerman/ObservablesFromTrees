import os, sys, json, shutil
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Fit transformer to data
def fit_transformer(allgraphs, featnames, save_path, transfname, plot=True):

    # Get all x data
    allx = np.zeros((1, len(featnames)))
    for i in range(len(allgraphs)):
        allx = np.vstack((allx, allgraphs[i].x.numpy()))

    # Fit and save transformer 
    catagoricalnames = ['num_prog(4)' 'mmp?(14)']
    catcols = [i for i in range(len(featnames)) if featnames[i] in catagoricalnames]
    tcols = [i for i in range(len(featnames)) if featnames[i] not in catagoricalnames]
    transf = get_transf(transfname)
    t = [('num', transf(output_distribution='normal'), tcols), ('cat', OneHotEncoder(), catcols)]
    transformer = ColumnTransformer(transformers=t)
    ft = transformer.fit(allx)

    # Plot
    if plot: 
        allx_t = ft.transform(allx)
        fig, ax = plt.subplots(nrows=7, ncols=7, figsize=(30,23)); ax = ax.flatten()
        for i in range(len(featnames)):
            ax[i].hist(allx[:,i], bins=50, density=1, histtype='step', color='b')
            ax2 = ax[i].twinx()
            ax2.hist(allx_t[:,i], bins=50, density=1, histtype='step', color='g')
            ax[i].set(title=featnames[i])
        fig.tight_layout()
        plt.savefig(save_path.replace('Data','Figs').replace('.pkl', '_hist.png'))
    
    with open(save_path, 'wb') as f:
        pickle.dump(ft, f)
    
    return ft


def get_transf(transf_name):

    import sklearn.preprocessing as skp

    transf = getattr(skp, transf_name)
    return transf

'''
NOT USED
'''

cwd = osp.abspath('')

def list_experiments(folder):
    experiment_folder = osp.join(cwd, folder, "todo") 
    experiment_files  = [file for file in os.listdir(experiment_folder) if file.endswith('json')] # in case of adtnl misc files
    return experiment_folder, experiment_files

#this doesn't work
def clean_done(folder):
    experiment_folder = osp.join(cwd, folder, "done") 
    legacy_path=osp.join(cwd, folder, 'legacy')
    if not osp.exists(legacy_path):
        os.mkdir(legacy_path)
        print('Legacy made')
    try:
        files  = os.listdir(experiment_folder)
        for f in files:
            shutil.move(osp.join(experiment_folder,f), folder, 'legacy/')
    except:
        os.mkdir(experiment_folder)
    print('Cleaned done folder')

##makes permutations of dict of lists very fast
def perms(diffs):
    from itertools import product
    keys=list(diffs.keys())
    val=list(diffs.values())
    for i, s in enumerate(val):
        if i==0:
            a=val[0]
        else:
            a=product(a, val[i])
    bs=[]
    for b in a:
        bs.append(b)
    output=[]
    def removeNestings(l):
        for i in l:
            if type(i) == tuple:
                removeNestings(i)
            else:
                output.append(i)
    removeNestings(bs)
    perms=np.array(output)
    return perms.reshape(-1, len(keys))
