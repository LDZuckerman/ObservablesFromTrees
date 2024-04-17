import os, sys, json, shutil, io
import os.path as osp
from matplotlib.font_manager import json_dump
from networkx import predecessor
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import torch
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader as DataLoader_notgeom
from torch_geometric.loader import DataLoader
from torch import nn
# sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
# from dev import data_utils
#import dev.data_utils as data_utils
try: 
    from dev import data_utils, models, loss_funcs
except:
    sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
    from ObservablesFromTrees.dev import data_utils, models, loss_funcs

# Fit transformer to data
def fit_transformer(allgraphs, featnames, save_path, transfname, plot=True, return_xt=False):

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
    
    if return_xt: 
        return ft, allx_t
    else: return ft


def get_transf(transf_name):

    import sklearn.preprocessing as skp

    transf = getattr(skp, transf_name)
    return transf

def get_model(model_name, hyper_params):
    '''
    Load model object from model folder
    '''

    sys.path.insert(0, 'ObservablesFromTrees/dev/') 
    from dev import models  # HOW COME THIS DOESNT WORK NOW THAT IVE MOVED THIS TO RUN_UTILS????

    model = getattr(models, model_name)
    model = model(**hyper_params)

    return model


def get_loss_func(name):
    '''
    Load loss function from loss_funcs file
    '''

    #import dev.loss_funcs as loss_funcs # HOW COME THIS DOESNT WORK NOW THAT IVE MOVED THIS TO RUN_UTILS????

    loss_func = getattr(loss_funcs, name)
    try:
        l=loss_func()
    except:
        l=loss_func

    return l


def test(loader, model, n_targ, get_var = False, return_x = False):
    '''
    Returns targets and predictions
    '''
    ys, yhats, vars, xs, = [],[], [], []  # add option to return xs
    model.eval()
    with torch.no_grad():
        for data in loader: 
            ys.append(data.y.view(-1,n_targ))
            xs.append(data.x)
            if get_var:
                pred_mu, pred_sig = model(data)
                yhats.append(pred_mu)
                vars.append(pred_sig)
            else:
                yhat = model(data)
                yhats.append(yhat)
    ys = torch.vstack(ys).cpu().numpy()
    yhats = torch.vstack(yhats).cpu().numpy()
    xs = torch.vstack(xs).cpu().numpy()

    if get_var:
        vars = torch.vstack(vars).cpu().numpy()
        preds = [yhats, vars]
    else:
        preds = yhats

    if return_x: 
        return ys, preds, xs
    else: 
        return ys, preds

def final_test(taskdir, modelname):
    '''
    Final test on test set
    '''

    sys.path.insert(0, 'ObservablesFromTrees/dev/') 
    import models, loss_funcs, data_utils

    # Load exp parameters
    config = json.load(open(os.path.join(taskdir, modelname, 'expfile.json'), 'rb'))
    run_params = config['run_params']
    hyper_params = config['hyper_params']
    data_params = config['data_params']
    used_targs = data_params['use_targs']
    used_feats = data_params['use_feats']
    hyper_params['get_sig'] = True if run_params['loss_func'] in ["GaussNd", "Navarro"] else False 
    hyper_params['get_cov'] = True if run_params['loss_func'] in ["GaussNd_corr"] else False

    # Load test data 
    hyper_params['in_channels'] = len(used_feats)
    hyper_params['out_channels'] = len(used_targs)
    test_data, _ = pickle.load(open(data_utils.get_subset_path(data_params, set = 'test'), 'rb')) 
    test_loader = DataLoader(test_data, batch_size=run_params['batch_size'], shuffle=0, num_workers=run_params['num_workers']) 
    
    # Run model on test data
    model = getattr(models, run_params['model'])(**hyper_params) 
    model.load_state_dict(torch.load(f'{taskdir}/{modelname}/model_best.pt', map_location=torch.device('cpu')))
    n_targ = len(used_targs)
    get_var = True if hyper_params['get_sig'] or hyper_params['get_cov'] else False 
    ys, preds = test(test_loader, model, n_targ, get_var) # testvars is [] if not get_var
    if get_var: yhats = preds[0]
    else: yhats = preds
    metrics = {'rmse': np.std(yhats - ys, axis=0),
               'rho': np.array([stats.pearsonr(ys[:,i], yhats[:,i]).statistic for i in range(ys.shape[1])]), # Pearson R
               'R2': np.array([skmetrics.r2_score(ys[:,i], yhats[:,i]) for i in range(ys.shape[1])]), # coefficient of determination
               'bias': np.mean(yhats - ys, axis=0)} # mean difference between yhat and y
    pickle.dump((ys, preds, metrics), open(f'{taskdir}/{modelname}/testres.pkl', 'wb'))

    return ys, preds, metrics


'''
For comparing to simple MLP
'''
def run_simplemodel(taskdir, predict_mu_sig, loss_fn, hidden_layers=1, n_epochs=100, lr=0.1, retrain=False, name=''):

    # Hardcoded parameters 
    datapath = '/mnt/home/lzuckerman/ceph/Data/vol100/'
    dataset = 'DS1'
    data_meta = json.load(open(f"{datapath}{dataset}_meta.json", 'rb'))
    all_labelnames = data_meta[1]['obs meta']['label_names'] 
    all_featnames = data_meta[2]['tree meta']['featnames']
    use_feats = ['#scale(0)', 'desc_scale(2)', 'Mvir(10)', 'Rvir(11)']

    # Create simple dataset (if not already created)
    if not os.path.exists(f'{datapath}{dataset}_finalhalosdata.pkl'):
        usetarg_idxs = [all_labelnames.index(targ) for targ in all_labelnames] 
        used_featidxs = [all_featnames.index(f) for f in all_featnames if f in use_feats]
        allgraphs = pickle.load(open(f"{datapath}{dataset}.pkl", 'rb'))
        traintransformer_path = osp.expanduser(f"{datapath}{dataset}_QuantileTransformer_train.pkl") 
        ft_train = pickle.load(open(traintransformer_path, 'rb')) # SHOULD REALLY BE USING SEPERATE TRANSFORMERS FOR TRAIN AND TEST BUT ACCIDENTALLY DIDNT SAVE TEST ONE YET
        sm_traindata, sm_testdata = data_utils.data_finalhalos(allgraphs, used_featidxs, usetarg_idxs, ft_train) # data_utils.data_fake()
        pickle.dump((sm_traindata, sm_testdata), open(f'{datapath}{dataset}_finalhalosdata.pkl', 'wb'))
    
    # Get data
    sm_traindata, sm_testdata = pickle.load(open(f'{datapath}{dataset}_finalhalosdata.pkl', 'rb'))
    sm_trainloader = DataLoader_notgeom(sm_traindata, batch_size=256, shuffle=True, num_workers=2)
    sm_testloader = DataLoader_notgeom(sm_testdata, batch_size=256, shuffle=True, num_workers=2)
    in_channels = len(next(iter(sm_traindata))[0])
    out_channels = len(next(iter(sm_traindata))[1])

    # Train (if not already trained)
    if predict_mu_sig:
        model = models.SimpleDistNet(hidden_layers, 50, in_channels, out_channels)
    else:
        model = models.SimpleNet(100, in_channels, out_channels)
    modelpath =  f'{taskdir}/MLP/{name}.pt'
    if (not os.path.exists(modelpath)) or retrain:
        loss_func = get_loss_func(loss_fn)
        simple_train(model, sm_trainloader, predict_mu_sig, loss_func, lr, n_epochs, name=name)
        torch.save(model.state_dict(), modelpath)

    # Test and save results
    model.load_state_dict(torch.load(modelpath))
    ys, preds = simple_test(sm_testloader, model, get_var = predict_mu_sig) 
    if predict_mu_sig: yhats = preds[0]
    else: yhats = preds
    metrics = {'rmse': np.std(yhats - ys, axis=0),
               'rho': np.array([stats.pearsonr(ys[:,i], yhats[:,i]).statistic for i in range(ys.shape[1])]), 
               'R2': np.array([skmetrics.r2_score(ys[:,i], yhats[:,i]) for i in range(ys.shape[1])]), 
               'bias': np.mean(yhats - ys, axis=0)}
    pickle.dump((ys, preds, metrics), open(f'{modelpath.replace(".pt", "")}_testres.pkl', 'wb'))

    # Save "config" file for future reference 
    config = {'run_params': {'model':name, 'loss_func':loss_fn, 'n_epochs':n_epochs}, 
              'hyper_params': {'hidden_layers':hidden_layers, 'lr':lr},
              'data_params': {'data_path':'~/ceph/Data/vol100/', 'data_file':'z0', 'use_feats':['#scale(0)', 'desc_scale(2)', 'Mvir(10)', 'Rvir(11)'], 'use_targs':['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']}}
    json.dump(config, open(f'{modelpath.replace(".pt", "")}_config.json', 'w'))

    return ys, preds, metrics

    
# Train simple model
def simple_train(model, trainloader, predict_mu_sig, loss_fn, lr, n_epochs=100, debug=False, name=''):
  
  print(f'Training {name}')
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  model.train()
  for e in range(n_epochs):
    tot_loss = 0
    i = 0 
    for x, y in trainloader:
      optimizer.zero_grad()
      if predict_mu_sig:
        if debug: print(f"{i}"); i+=1
        muhat, sighat = model(x)
        loss, _, _ = loss_fn(muhat, y, sighat)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optimizer.step()
        tot_loss += loss.item()
      else: 
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    #print(f'[Epoch {e + 1}] avg loss: {np.round(tot_loss/len(trainloader),5)}')
    print(f'  Epoch {e + 1}', end="\r")

# Test simple model
def simple_test(loader, model, get_var):
    '''
    Returns targets and predictions
    '''

    print('Testing')
    ys, yhats, vars, xs, = [],[], [], []  # add option to return xs
    model.eval()
    with torch.no_grad():
        for x, y in loader: 
            ys.append(y)
            xs.append(x)
            if get_var:
                pred_mu, pred_sig = model(x)
                yhats.append(pred_mu)
                vars.append(pred_sig)
            else:
                yhat = model(x)
                yhats.append(yhat)
    ys = torch.vstack(ys).cpu().numpy()
    yhats = torch.vstack(yhats).cpu().numpy()
    xs = torch.vstack(xs).cpu().numpy()

    if get_var:
        vars = torch.vstack(vars).cpu().numpy()
        preds = [yhats, vars]
    else:
        preds = yhats

    return ys, preds

# # Test simple model
# def simple_test(model, testloader, predict_mu_sig=False):

#     model.eval()

#     if predict_mu_sig:

#         with torch.no_grad():
#             predmus = []
#             predsigs = []
#             ys = []
#             for x, y in testloader:
#                 muhat, sighat = model(x)
#                 predmus.append(muhat)
#                 predsigs.append(sighat)
#                 ys.append(y)
#             predmus = torch.cat(predmus, dim=0).detach().numpy()
#             predsigs = torch.cat(predsigs, dim=0).detach().numpy()
#             ys = torch.cat(ys, dim=0).detach().numpy()
        
#         return ys, predmus, predsigs
    
#     else: 

#         with torch.no_grad():
#             preds = []
#             ys = []
#             for x, y in testloader:
#                 yhat = model(x)
#                 preds.append(yhat)
#                 ys.append(y)
#             preds = torch.cat(preds, dim=0).detach().numpy()
#             ys = torch.cat(ys, dim=0).detach().numpy()
        
#         return ys, preds

'''
This should really go somewhere else
'''

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def get_modelsDF(taskdir):

    # Create dict
    models_list = ['MLP1', 'MLP2', 'MLP3'] + [exp for exp in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir, exp)) and exp.startswith('e') and exp not in ['exp_tiny', 'exp1']]
    cols = ['exp', 'test rho', 'test rmse', 'test R2', 'test bias', 'val rmse', 'base', 'loss func', 'n epochs', 'stop epoch', 'feats', 'targs', 'DS']
    models = pd.DataFrame(columns = cols)
    
    # Fill with models 
    for model in models_list:
        if 'MLP' in model: 
            resfile = f'{taskdir}/MLP/{model}_testres.pkl'
            config = json.load(open(f'{taskdir}/MLP/{model}_config.json', 'rb'))
        else: 
            resfile = osp.join(taskdir, model, 'testres.pkl')
            config = json.load(open(osp.join(taskdir, model, 'expfile.json'), 'rb'))
            try: 
                train_results = CPU_Unpickler(open(osp.join(taskdir, model, 'result_dict.pkl'), 'rb')).load()
            except FileNotFoundError:
                print(f"Ignoreing {model} (has no result_dict.pkl)")
                continue
        _, _, test_results = pickle.load(open(resfile, 'rb'))
        vol = config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        ds = config['data_params']['data_file'].replace('.pkl','')
        info = [model,
                np.round(np.mean(test_results['rho']),2), # mean over all targets
                np.round(np.mean(test_results['rmse']),2), # mean over all targets
                np.round(np.mean(test_results['R2']),2), # mean over all targets
                np.round(np.mean(test_results['bias']),2), # mean over all targets
                np.round(np.mean(train_results['val rmse final test']),2) if not 'MLP' in model else '',
                config['run_params']['model'], 
                config['run_params']['loss_func'], 
                config['run_params']['n_epochs'], 
                train_results['epochexit'] if not 'MLP' in model else '', 
                config['data_params']['use_feats'], 
                config['data_params']['use_targs'],
               f'{vol}, {ds}']
        models.loc[len(models)] = info
    models = models.sort_values(by=['test rho', 'test rmse', 'test R2', 'val rmse'], ascending=[False, True, False, True])

    return models


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