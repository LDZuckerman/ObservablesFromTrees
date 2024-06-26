import os, sys, json, shutil, io
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from sympy import E
import torch
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader as DataLoader_notgeom
from torch_geometric.loader import DataLoader
from torch import nn
import sklearn.metrics as skmetrics
try: 
    from dev import data_utils, models, train_script, loss_funcs
except ModuleNotFoundError:
    try:
        sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
        from ObservablesFromTrees.dev import data_utils, models, train_script, loss_funcs
    except ModuleNotFoundError:
        sys.path.insert(0, 'ObservablesFromTrees/dev/')
        from dev import data_utils, models, train_script, loss_funcs

# Fit transformer to data
def fit_transformer(allgraphs, featnames, save_path, transfname, plot=False, return_xt=False):

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
    from dev import models 

    model = getattr(models, model_name)
    model = model(**hyper_params)

    return model


def get_loss_func(loss_name):
    '''
    Load loss function from loss_funcs file
    '''
    # sys.path.insert(0, 'ObservablesFromTrees/dev/') 
    # from dev import loss_funcs

    loss_func = getattr(loss_funcs, loss_name)
    try:
        l=loss_func()
    except:
        l=loss_func

    return l

def get_act_func(act_name):
    '''
    Load activation function from torch
    '''
    #act_func = getattr(nn, act_name)
    if act_name =='ReLU':
        act_func = nn.ReLU()
    elif act_name =='LeakyReLU': 
        act_func= nn.LeakyReLU()
    elif act_name =='Sigmoid':
        act_func = nn.Sigmoid()
    elif act_name =='Tanh':
        act_func = nn.Tanh()
    elif act_name =='Softmax':
        act_func = nn.Softmax()
    else: raise ValueError(f'Activation function {act_name} not recognized. Choose from ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, or add to get_act_func in run_utils.py.')

    return act_func


def get_downsize_func(method_num):

    func_name = 'downsize_tree'+str(method_num)
    function = getattr(data_utils, func_name)
    
    return function


def test(loader, model, n_targ, predict_dist = False, return_x = False):
    '''
    Returns targets and predictions
    '''
    ys, yhats, sigs, xs, = [],[], [], []  # add option to return xs
    model.eval()
    with torch.no_grad():
        for data in loader: 
            ys.append(data.y.view(-1,n_targ))
            xs.append(data.x)
            if predict_dist:
                pred_mu, pred_logsig = model(data)
                yhats.append(pred_mu)
                sigs.append(torch.exp(pred_logsig)) # convert from log(sig) to sig
            else:
                yhat = model(data)
                yhats.append(yhat)
    ys = torch.vstack(ys).cpu().numpy()
    yhats = torch.vstack(yhats).cpu().numpy()
    xs = torch.vstack(xs).cpu().numpy()

    if predict_dist:
        sigs = torch.vstack(sigs).cpu().numpy()
        preds = [yhats, sigs]
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
    import models, data_utils

    # Load exp parameters
    config = json.load(open(os.path.join(taskdir, modelname, 'expfile.json'), 'rb'))
    run_params = config['run_params']
    hyper_params = config['hyper_params']
    data_params = config['data_params']
    used_targs = data_params['use_targs']
    used_feats = data_params['use_feats']
    hyper_params['get_sig'] = True if run_params['loss_func'] in ["GaussNd", "Navarro", "GaussNd_torch"] else False 
    hyper_params['get_cov'] = True if run_params['loss_func'] in ["GaussNd_corr"] else False

    # Load test data 
    hyper_params['in_channels'] = len(used_feats)
    hyper_params['out_channels'] = len(used_targs)
    if 'sam' not in data_params['data_path']:
        test_data, _ = pickle.load(open(data_utils.get_subset_path(data_params, set = 'test')  , 'rb'))
    else:
        test_data = pickle.load(open(osp.expanduser(f"{data_params['data_path']}/{data_params['data_file']}".replace('.pkl', '_testdata.pkl'))  , 'rb'))
    test_loader = DataLoader(test_data, batch_size=run_params['batch_size'], shuffle=0, num_workers=run_params['num_workers']) 
    
    # Run model on test data
    model = getattr(models, run_params['model'])(**hyper_params) 
    model.load_state_dict(torch.load(f'{taskdir}/{modelname}/model_best.pt', map_location=torch.device('cpu')))
    n_targ = len(used_targs)
    predict_dist = True if hyper_params['get_sig'] or hyper_params['get_cov'] else False 
    ys, preds = test(test_loader, model, n_targ, predict_dist) # returned preds[1] is sig not logsig
    if predict_dist: yhats = preds[0]
    else: yhats = preds
    metrics = {'rmse': np.std(yhats - ys, axis=0),
               'rho': np.array([stats.pearsonr(ys[:,i], yhats[:,i]).statistic for i in range(ys.shape[1])]), # Pearson R
               'R2': np.array([skmetrics.r2_score(ys[:,i], yhats[:,i]) for i in range(ys.shape[1])]), # coefficient of determination
               'bias': np.mean(yhats - ys, axis=0)} # mean difference between yhat and y
    print(f"  Saving test results for {modelname} to {taskdir}/{modelname}/testres.pkl")
    pickle.dump((ys, preds, metrics), open(f'{taskdir}/{modelname}/testres.pkl', 'wb'))

    return ys, preds, metrics


# def final_test(taskdir, modelname):
#     '''
#     Final test on test set
#     '''

#     sys.path.insert(0, 'ObservablesFromTrees/dev/') 
#     import models, loss_funcs, data_utils

#     # Load exp parameters
#     config = json.load(open(os.path.join(taskdir, modelname, 'expfile.json'), 'rb'))
#     run_params = config['run_params']
#     hyper_params = config['hyper_params']
#     data_params = config['data_params']
#     used_targs = data_params['use_targs']
#     used_feats = data_params['use_feats']

#     # Load test data 
#     hyper_params['in_channels'] = len(used_feats)
#     hyper_params['out_channels'] = len(used_targs)
#     test_data, _ = pickle.load(open(data_utils.get_subset_path(data_params, set = 'test'), 'rb')) 
#     test_loader = DataLoader(test_data, batch_size=run_params['batch_size'], shuffle=0, num_workers=run_params['num_workers']) 
    
#     # Run model on test data
#     model = getattr(models, run_params['model'])(**hyper_params)  #  = run_utils.get_model(run_params['model'], hyper_params)
#     model.load_state_dict(torch.load(f'{taskdir}/{modelname}/model_best.pt', map_location=torch.device('cpu')))
#     loss_func = getattr(loss_funcs, run_params['loss_func'])  # loss_func = run_utils.get_loss_func(run_params['loss_func'])
#     get_var = True if run_params['loss_func'] in ["Gauss1d", "Gauss2d", "GaussNd", "Gauss2d_corr", "Gauss4d_corr"] else False
#     get_rho = True if run_params['loss_func'] in ["Gauss2d_corr", "Gauss4d_corr"] else False 
#     n_targ = len(used_targs)
#     testys, testpreds, _ = test(test_loader, model, n_targ, get_var, get_rho) 
#     metrics = {'sigma': np.std(testpreds - testys, axis=0),
#                'R': np.array([stats.pearsonr(testys[:,i], testpreds[:,i]).statistic for i in range(testys.shape[1])])}
#     pickle.dump((testys, testpreds, metrics), open(f'{taskdir}/{modelname}/testres.pkl', 'wb'))

#     return testys, testpreds, metrics



# ''' HOW DID I END UP WITH THESE VERSIONS?? MAIN HAS THE UPDATED ONES!!! DID I FORGET TO PULL DOWN MAIN BEFORE MAKING NEW BRANCH? CRAP. WHAT ELSE IS AND OLS VERSION???
# For comparing to simple MLP
# '''
# def run_simplemodel(taskdir):

#     # Hardcoded parameters 
#     datapath = '/mnt/home/lzuckerman/ceph/Data/vol100/'
#     dataset = 'DS1'
#     data_meta = json.load(open(f"{datapath}{dataset}_meta.json", 'rb'))
#     all_labelnames = data_meta[1]['obs meta']['label_names'] 
#     all_featnames = data_meta[2]['tree meta']['featnames']
#     use_feats = ['#scale(0)', 'desc_scale(2)', 'Mvir(10)', 'Rvir(11)']

#     # Train (if not already trained)
#     if not osp.exists(f'{taskdir}/model_justfinalhalos.pt'):

#         # Create simple dataset (if not already created)
#         if not os.path.exists(f'{datapath}{dataset}_finalhalosdata.pkl'):
#             usetarg_idxs = [all_labelnames.index(targ) for targ in all_labelnames] 
#             used_featidxs = [all_featnames.index(f) for f in all_featnames if f in use_feats]
#             allgraphs = pickle.load(open(f"{datapath}{dataset}.pkl", 'rb'))
#             traintransformer_path = osp.expanduser(f"{datapath}{dataset}_QuantileTransformer_train.pkl") 
#             ft_train = pickle.load(open(traintransformer_path, 'rb')) # SHOULD REALLY BE USING SEPERATE TRANSFORMERS FOR TRAIN AND TEST BUT ACCIDENTALLY DIDNT SAVE TEST ONE YET
#             sm_traindata, sm_testdata = data_utils.data_finalhalos(allgraphs, used_featidxs, usetarg_idxs, ft_train) # data_utils.data_fake()
#             pickle.dump((sm_traindata, sm_testdata), open(f'{datapath}{dataset}_finalhalosdata.pkl', 'wb'))
        
#         # Train simple model
#         sm_traindata, sm_testdata = pickle.load(open(f'{datapath}{dataset}_finalhalosdata.pkl', 'rb'))
#         sm_trainloader = DataLoader_notgeom(sm_traindata, batch_size=256, shuffle=True, num_workers=2)
#         sm_testloader = DataLoader_notgeom(sm_testdata, batch_size=256, shuffle=True, num_workers=2)
#         in_channels = len(next(iter(sm_traindata))[0])
#         out_channels = len(next(iter(sm_traindata))[1])
#         sm_model = models.SimpleNet(100, in_channels, out_channels)
#         simple_train(sm_model, sm_trainloader, n_epochs = 100, loss_fn = nn.MSELoss())
#         torch.save(sm_model.state_dict(), f'{taskdir}/model_justfinalhalos.pt')

#     # Test
#     sm_model = models.SimpleNet(100, in_channels, out_channels)
#     sm_model.load_state_dict(torch.load(f'{taskdir}/model_justfinalhalos.pt'))
#     sm_testys, sm_testpreds = simple_test(sm_model, sm_testloader)
#     sm_metrics = {'sigma': np.std(sm_testpreds - sm_testys, axis=0),
#                 'R': np.array([stats.pearsonr(sm_testys[:,i], sm_testpreds[:,i]).statistic for i in range(sm_testys.shape[1])])}
#     pickle.dump((sm_testys, sm_testpreds, sm_metrics), open(f'{taskdir}/model_justfinalhalos_testres.pkl', 'wb'))

# # Train simple model
# def simple_train(model, trainloader, n_epochs, loss_fn):
  
#   optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#   model.train()
#   for e in range(n_epochs):
#     tot_loss = 0
#     for x, y in trainloader:
#       optimizer.zero_grad()
#       yhat = model(x)
#       loss = loss_fn(yhat, y)
#       loss.backward()
#       optimizer.step()
#       tot_loss += loss.item()

#     print(f'[Epoch {e + 1}] avg loss: {np.round(tot_loss/len(trainloader),5)}')

# # Test simple model
# def simple_test(model, testloader):

#     model.eval()
#     with torch.no_grad():
#         preds = []
#         ys = []
#         for x, y in testloader:
#             yhat = model(x)
#             preds.append(yhat)
#             ys.append(y)
#         preds = torch.cat(preds, dim=0)
#         ys = torch.cat(ys, dim=0)
#         preds = preds.detach().numpy()
#         ys = ys.detach().numpy()
    
#     return ys, preds

'''
This should really go somewhere else
'''

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_modelResultsDF(taskdir, dataset=None):

    # Get list of models (experiments) and FHO models
    models_list = [exp for exp in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir, exp)) and exp.startswith('e') and exp not in ['exp_tiny']]
    models_list += [exp.replace('.pt','') for exp in os.listdir(f"{taskdir}/FHO/") if exp.endswith('.pt')]
    all_info = []
    
    # Create dict of model information (for models with desired dataset)
    for model in models_list:

        # Get config, train results, and test results files
        if 'FHO' in model: 
            trnresfile = f'{taskdir}/FHO/{model}_train_results.pkl'
            testresfile = f'{taskdir}/FHO/{model}_testres.pkl'
            config = json.load(open(f'{taskdir}/FHO/{model}_config.json', 'rb'))
        else: 
            trnresfile = osp.join(taskdir, model, 'result_dict.pkl')
            testresfile = osp.join(taskdir, model, 'testres.pkl')
            config = json.load(open(osp.join(taskdir, model, 'expfile.json'), 'rb'))

        # If this is a completed model, load results, check if only pt-estimating, and re-compute metrics for a bunch of predicted distribution samples
        if not osp.exists(testresfile):
            print(f"Skipping {model} no test results file found")
            continue
        testys, testpreds, test_results = pickle.load(open(testresfile, 'rb'))
        if len(testpreds) != 2: 
            print(f"Skipping {model} (point est only)")
            continue
        rhos, rmses, R2s, baises, scatters = data_utils.sample_results(testys, testpreds, n_samples=100)
        if np.any(np.mean(rmses) > 1000): 
            print(f"Skipping {model} (ridiculously high RMSE)")
            continue

        # If this model has a training results file (wasnt saving for early FHOs), load it and get stop epoch
        if osp.exists(trnresfile):
            try: trnres = pickle.load(open(trnresfile, 'rb'))
            except RuntimeError: 
                trnres = data_utils.CPU_Unpickler(open(trnresfile, 'rb')).load()
            try: epochs = f"{trnres['epochexit']} (of {config['run_params']['n_epochs']})" 
            except KeyError: epochs = config['run_params']['n_epochs']
        else: epochs = config['run_params']['n_epochs']

        # Get means and SDs of each metric over all samples
        mean_rho = np.mean(rhos, axis=0) 
        sd_rho = np.std(rhos, axis=0) 
        mean_rmse = np.mean(rmses, axis=0) 
        sd_rmse = np.std(rmses, axis=0) 
        mean_R2 = np.mean(R2s, axis=0)
        sd_R2 = np.std(R2s, axis=0)
        mean_bias = np.mean(baises, axis=0)
        sd_bias = np.std(baises, axis=0)
        mean_scatter = np.mean(scatters, axis=0)
        sd_scatter = np.std(scatters, axis=0)

        # Add to dict the means of the means and SDs over all targets
        datafile = config['data_params']['data_file']
        if dataset != None: # for any tasks that arent sam props I'll want to look at models for a sepcific dataset
            ds = datafile[:3]
            if ds != dataset:
                continue
            vol = 'SAM' if 'FHO' in model else config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        else:
            ds = datafile.replace('.pkl','')
            vol = 'SAM' if 'FHO' in model else config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        pm = r'$\pm$'
        info = {'exp': model, 
                #'rho': np.round(np.mean(test_results['rho']),2), # mean over all targets
                'mean rho': f"{np.round(np.mean(mean_rho),2)} +/- {np.round(np.mean(sd_rho),3)}",  # mean over all targets
                #'rmse': np.round(np.mean(test_results['rmse']),2), # mean over all targets
                'mean rmse': f"{np.round(np.mean(mean_rmse),2)} +/- {np.round(np.mean(sd_rmse),3)}", # mean over all targets
                #'R2': np.round(np.mean(test_results['R2']),2), # mean over all targets, 
                'mean R2': f"{np.round(np.mean(mean_R2),2)} +/- {np.round(np.mean(sd_R2),3)}",  # mean over all targets
                #'bias': np.round(np.mean(test_results['bias']),2), # mean over all targets
                'mean bais': f"{np.round(np.mean(mean_bias),2)} +/- {np.round(np.mean(sd_bias),3)}",  # mean over all targets
                #'val rmse': np.round(np.mean(train_results['val rmse final test']),2) if not 'FHO' in model else '',
                'mean sactter': f"{np.round(np.mean(mean_scatter),2)} +/- {np.round(np.mean(sd_scatter),3)}",  # mean over all targets 
                'loss func': config['run_params']['loss_func'], 
                'n epochs': epochs, 
                'feats' : config['data_params']['use_feats'], 
                #'targs': config['data_params']['use_targs'], 
                'DS': f'{vol}, {ds}'
                }
        all_info.append(info)

    # Make into DF and sort
    models = pd.DataFrame.from_dict(all_info)   
    #models['abs bias'] = np.abs(models['bias']) 
    #models = models.sort_values(by=['rho', 'rmse', 'R2', 'abs bias'], ascending=[False, True, False, False])
    #models.drop(columns=['abs bias'], inplace=True)


    return models

def get_modelsDF(taskdir, type='GNN', dataset=None):

    # Get list of models (experiments) and FHO models
    if type == 'GNN':
        models_list = [exp for exp in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir, exp)) and exp.startswith('e') and exp not in ['exp_tiny']]
    elif type == 'MLP':
        models_list = [exp.replace('.pt','') for exp in os.listdir(f"{taskdir}/FHO/") if exp.endswith('.pt')]
    else: raise ValueError(f"Type {type} not recognized. Choose from 'GNN' or 'MLP'")
    all_info = []
    
    # Create dict of model information (for models with desired dataset)
    for model in models_list:

        # Get config, train results, and test results files
        if type == 'MLP': 
            trnresfile = f'{taskdir}/FHO/{model}_train_results.pkl'
            testresfile = f'{taskdir}/FHO/{model}_testres.pkl'
            config = json.load(open(f'{taskdir}/FHO/{model}_config.json', 'rb'))
            result_notes = json.load(open('Task_phot/FHO/all_FHO_res_notes.json','rb'))
        else: 
            trnresfile = osp.join(taskdir, model, 'result_dict.pkl')
            testresfile = osp.join(taskdir, model, 'testres.pkl')
            config = json.load(open(osp.join(taskdir, model, 'expfile.json'), 'rb'))

        # If this is a completed model, load results, check if only pt-estimating, and re-compute metrics for a bunch of predicted distribution samples
        if not osp.exists(testresfile):
            print(f"Skipping {model} no test results file found")
            continue
        testys, testpreds, test_results = pickle.load(open(testresfile, 'rb'))
        if len(testpreds) != 2: 
            print(f"Skipping {model} (point est only)")
            continue

        # If this model has a training results file (wasnt saving for early FHOs), load it and get stop epoch
        if osp.exists(trnresfile):
            try: trnres = pickle.load(open(trnresfile, 'rb'))
            except RuntimeError: 
                trnres = data_utils.CPU_Unpickler(open(trnresfile, 'rb')).load()
            try: epochs = f"{trnres['epochexit']} (of {config['run_params']['n_epochs']})" 
            except KeyError: epochs = config['run_params']['n_epochs']
        else: epochs = config['run_params']['n_epochs']

        # Add to dict the means of the means and SDs over all targets
        datafile = config['data_params']['data_file']
        if dataset != None: # for any tasks that arent sam props I'll want to look at models for a sepcific dataset
            ds = datafile[:3]
            if ds != dataset:
                continue
            vol = 'vol100' if 'data_path' not in config['data_params'].keys() else config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        else:
            ds = datafile.replace('.pkl','')
            vol = 'SAM' if 'FHO' in model else config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        pm = r'$\pm$'
        if type == 'MLP':
            info = {'exp': model, 
                    'modelname': 'SimpleDistNet' if 'modelname' not in config.keys() else config['modelname'],
                    'loss func': config['run_params']['loss_func'], 
                    'clip': 'True' if 'learn_params' not in config.keys() or 'clip' not in config['learn_params'] else config['learn_params']['clip'],
                    'HLs': '' if 'hidden_layers' not in config['hyper_params'].keys() else config['hyper_params']['hidden_layers'],
                    'HCs': '' if 'hidden_channels' not in config['hyper_params'].keys() else config['hyper_params']['hidden_channels'],
                    'activation': '' if 'activation' not in config['hyper_params'].keys() else config['hyper_params']['activation'],
                    'LN': 'True' if 'layer_norm' not in config['hyper_params'].keys() else config['hyper_params']['layer_norm'],
                    'n epochs': epochs, 
                    'feats' : config['data_params']['use_feats'], 
                    #'targs': config['data_params']['use_targs'], 
                    'DS': f'{vol}, {ds}',
                    'note': '' if model not in result_notes.keys() else result_notes[model],
                    }
        else:
            info = {'exp': model, 
                    'n epochs': epochs, 
                    'feats' : config['data_params']['use_feats'], 
                    #'targs': config['data_params']['use_targs'], 
                    'DS': f'{vol}, {ds}'
                    }
        all_info.append(info)

    # Make into DF and sort
    models = pd.DataFrame.from_dict(all_info)   
    #models['abs bias'] = np.abs(models['bias']) 
    #models = models.sort_values(by=['rho', 'rmse', 'R2', 'abs bias'], ascending=[False, True, False, False])
    #models.drop(columns=['abs bias'], inplace=True)


    return models


# def get_modelsDF(taskdir):

#     # Create dict
#     models_list = [exp for exp in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir, exp)) and exp.startswith('e') and exp not in ['exp_tiny', 'exp1']]
#     cols = ['exp', 'DS', 'avg val sig', 'avg test sig', 'base', 'loss func', 'n epochs', 'stop epoch', 'feats', 'targs']
#     models = pd.DataFrame(columns = cols)

#     # Add final halo only model for comparison
#     _, _, sm_test_results = pickle.load(open(osp.join(taskdir, 'model_justfinalhalos_testres.pkl'), 'rb'))
#     info_sm = ['MLP', 'vol100, z0', '',
#                np.round(np.mean(sm_test_results['sigma']),2), # mean over all targets
#                '', '', '', '',
#                '[#scale(0), desc_scale(2), Mvir(10), Rvir(11)]', '[U, B, V, K, g, r, i, z]']
#     models.loc[0] = info_sm

#     # Add all other models
#     for model in models_list:
#         try: config = json.load(open(osp.join(taskdir, model, 'expfile.json'), 'rb'))
#         except FileNotFoundError:
#             print(f"ignoreing {model} (has no expfile.json)")
#             continue
#         try: results = CPU_Unpickler(open(osp.join(taskdir, model, 'result_dict.pkl'), 'rb')).load()
#         except FileNotFoundError:
#             print(f"Ignoreing {model} (has no result_dict.pkl)")
#             continue
#         try: 
#             _, _, test_results = pickle.load(open(osp.join(taskdir, model, 'testres.pkl'), 'rb'))
#             test_sig = np.round(np.mean(test_results['sigma']), 2) # mean over all targets
#         except FileNotFoundError:
#             test_sig = ''
#         vol = config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
#         ds = config['data_params']['data_file'].replace('.pkl','')
#         info = [model,
#                 f'{vol}, {ds}',
#                 np.round(np.mean(results['val sig final test']),2), # mean over all targets
#                 test_sig,
#                 config['run_params']['model'], 
#                 config['run_params']['loss_func'], 
#                 config['run_params']['n_epochs'], 
#                 results['epochexit'], 
#                 config['data_params']['use_feats'], 
#                 config['data_params']['use_targs']]
#         models.loc[len(models)] = info

#     return models


# Perhaps this should go in loss_funcs at some point, if I use it as one

def MMD(x, y, kernel):
    """
    This function is from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html

    Emprical maximum mean discrepancy. 
    The lower the result the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.array(x))
    if not isinstance(y, torch.Tensor): 
        y = torch.tensor(np.array(y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf": # guassian?

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


'''
For runnng sweeps
'''

def run_sweep_proc(paramdicts, sweep_dir, proc_num):

    # Loop over param sets
    count = 0
    print(f"[proc {proc_num}] Starting loop over {len(paramdicts)} parameter sets.")     
    for d in paramdicts:

        # # Create downsized dict that will be what is saved to done experiments file
        # save_dict = {}
        # save_dict['run_params'] = dict['run_params']
        # save_dict['learn_params'] = dict['learn_params']
        # save_dict['hyper_params'] = dict['hyper_params']

        # Check if this experiment has already been done
        exp_id = d['exp_id']
        done_exps = [f.replace('_model_best.pt','') for f in os.listdir(sweep_dir) if f.endswith('model_best.pt')]
        if exp_id in done_exps:
            print(f"[proc {proc_num}] Skipping experiment {exp_id} ({exp_id}_model_best.pt already exists in {sweep_dir})")
            continue
        
        # Set unique experiment ID
        print(f"[proc {proc_num}] SETTING UP EXPERIMENT {exp_id} with the following parameters -> {d}")

        # Train and save trained model results to temporary output directory 
        try:
            train_script.run(d['run_params'], d['data_params'], d['learn_params'], d['hyper_params'], sweep_dir, exp_id, gpu=True, low_output=True, proc=proc_num)
        except Exception as e:
            return e
        count += 1
        
        # # Save experiment results dict to this thread's temporary results file
        # if not osp.exists(results_file_proc):
        #     pickle.dump({expid: result_dict}, open(results_file_proc, 'wb'))
        # results_proc = pickle.load(open(results_file_proc, 'rb'))
        # results_proc.update({expid: result_dict})
        # pickle.dump(results_proc, open(results_proc, 'wb')) # open file twice to ensure overwrite
       
        # # Save experiment config dict to this thread's temporary done experiments file
        # if not osp.exists(configs_file_proc):
        #     json.dump({expid: save_dict}, open(configs_file_proc, 'wb'))
        # done_exps_proc = json.load(open(configs_file_proc, 'rb'))
        # done_exps_proc.update({expid: save_dict})
        # json.dump(done_exps_proc, open(configs_file_proc, 'wb')) # open file twice to ensure overwrite
        print(f'[proc {proc_num}] Done with {exp_id} [{count}/{len(paramdicts)}]\n')

    print(f"[proc {proc_num}] Done with all {len(paramdicts)} parameter sets.")

    


# def move_thread_outputs(procs_outdir, sweep_dir):
#     '''
#     Move the configs, trained modls, and results files from the temporary threads directory into the sweep folder
#     Replace the exp ids with the next available unique id
#     '''

#     print(f"Collecting experiment dicts, trained models, and results from all threads in {procs_outdir} and moving to {sweep_dir}", flush=True)

#     # If there are already some done experiments, find the next exp number
#     used_exp_ids = [f.replace('model_best.pt','')for f in os.listdir(sweep_dir) if f.endswith('model_best.pt')]
#     used_exp_nums = [int(id.replace('exp','')) for id in used_exp_ids]
#     if len(used_exp_nums) > 0:
#         new_num = "{:03d}".format(np.max(used_exp_nums) + 1) 
#     else:
#         new_num = '000'

#     # Loop over all experiments done by this sweep, find new unique exp id, and move the config, trained model, and results files out of the procs_temp directory
#     swept_exp_ids = [f.replace('model_best.pt','') for f in os.listdir(procs_outdir) if f.endswith('model_best.pt')]
#     for exp_id in swept_exp_ids:
#         new_id = f"exp{new_num}"
#         shutil.copy(f"{procs_outdir}/{exp_id}_model_best.pt", f"{sweep_dir}/{new_id}_model_best.pt")
#         shutil.copy(f"{procs_outdir}/{exp_id}_config.json", f"{sweep_dir}/{new_id}_config.json")
#         shutil.copy(f"{procs_outdir}/{exp_id}_result_dict.pkl", f"{sweep_dir}/{new_id}_result_dict.pkl")
#         new_num = "{:03d}".format(int(new_num) + 1) # increment new_num

#     ## Delete temporary procs_outdir
#     # os.remove(procs_outdir)

#     return 

# def get_swp_num(configs_file):

#     done_exps = json.load(open(configs_file, 'rb')) # dict of {exp_id: {dict}} for all done experiments
#     used_nums =  [int(n[-3:]) for n in list(done_exps.keys())]  # all numeric ids already created for this sweep
#     if len(used_nums)==0:
#         swp_num = '000'
#     else:
#         swp_num = "{:03d}".format(np.max(used_nums) + 1) 
    
#     return swp_num

# def make_info_dict(basedict, diffdict):

#     info_dict = basedict.copy()
#     for pkey in basedict.keys():
#         for key in basedict[pkey].keys():
#             info_dict[pkey][key] = diffdict[key]
            
#     return info_dict

# def check_done(save_dict, configs_file):

#     if not osp.exists(configs_file): # if configs file not yet create, no experiments have been done
#         done = False
#     else:
#         done_exps = json.load(open(configs_file, 'rb'))
#         done = save_dict in list(done_exps.values()) # done_exps.values() is list of exp dicts

#     return done

def perms(diffs):
    '''
    From Christian
    Makes permutations of dict of lists very fast
    '''
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

