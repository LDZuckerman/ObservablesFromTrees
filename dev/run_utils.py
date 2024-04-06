import os, sys, json, shutil, io
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
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

    import dev.loss_funcs as loss_func_module  # HOW COME THIS DOESNT WORK NOW THAT IVE MOVED THIS TO RUN_UTILS????

    loss_func = getattr(loss_func_module, name)
    try:
        l=loss_func()
    except:
        l=loss_func

    return l


def test(loader, model, n_targ, get_var = False, get_rho = False, return_x = False):
    '''
    Returns targets and predictions
    '''
    ys, preds, xs, vars, rhos = [],[], [], [], [] # add option to return xs
    model.eval()
    with torch.no_grad():
        for data in loader: 
            rho = torch.IntTensor(0)
            var = torch.IntTensor(0)
            if get_var and get_rho:
                out, var, rho = model(data)
            elif get_var:
                out, var = model(data)
            else:
                out = model(data)
            ys.append(data.y.view(-1,n_targ))
            preds.append(out)
            xs.append(data.x)
            vars.append(var)
            rhos.append(rho)
    ys = torch.vstack(ys)
    preds = torch.vstack(preds)
    xs = torch.vstack(xs)
    vars = torch.vstack(vars)
    rhos = torch.vstack(rhos)

    if return_x: 
        if get_rho: 
            return ys.cpu().numpy(), preds.cpu().numpy(), xs.cpu().numpy(), vars, rhos
        else:
            return ys.cpu().numpy(), preds.cpu().numpy(), xs.cpu().numpy(), vars
    else: 
        if get_rho: 
            return ys.cpu().numpy(), preds.cpu().numpy(), vars, rhos
        else:
            return ys.cpu().numpy(), preds.cpu().numpy(), vars

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

    # Load test data 
    hyper_params['in_channels'] = len(used_feats)
    hyper_params['out_channels'] = len(used_targs)
    test_data, _ = pickle.load(open(data_utils.get_subset_path(data_params, set = 'test'), 'rb')) 
    test_loader = DataLoader(test_data, batch_size=run_params['batch_size'], shuffle=0, num_workers=run_params['num_workers']) 
    
    # Run model on test data
    model = getattr(models, run_params['model'])(**hyper_params)  #  = run_utils.get_model(run_params['model'], hyper_params)
    model.load_state_dict(torch.load(f'{taskdir}/{modelname}/model_best.pt', map_location=torch.device('cpu')))
    loss_func = getattr(loss_funcs, run_params['loss_func'])  # loss_func = run_utils.get_loss_func(run_params['loss_func'])
    get_var = True if run_params['loss_func'] in ["Gauss1d", "Gauss2d", "GaussNd", "Gauss2d_corr", "Gauss4d_corr"] else False
    get_rho = True if run_params['loss_func'] in ["Gauss2d_corr", "Gauss4d_corr"] else False 
    n_targ = len(used_targs)
    testys, testpreds, _ = test(test_loader, model, n_targ, get_var, get_rho) 
    metrics = {'sigma': np.std(testpreds - testys, axis=0),
               'R': np.array([stats.pearsonr(testys[:,i], testpreds[:,i]).statistic for i in range(testys.shape[1])])}
    pickle.dump((testys, testpreds, metrics), open(f'{taskdir}/{modelname}/testres.pkl', 'wb'))

    return testys, testpreds, metrics



'''
For comparing to simple MLP
'''
def run_simplemodel(taskdir):

    # Hardcoded parameters 
    datapath = '/mnt/home/lzuckerman/ceph/Data/vol100/'
    dataset = 'DS1'
    data_meta = json.load(open(f"{datapath}{dataset}_meta.json", 'rb'))
    all_labelnames = data_meta[1]['obs meta']['label_names'] 
    all_featnames = data_meta[2]['tree meta']['featnames']
    use_feats = ['#scale(0)', 'desc_scale(2)', 'Mvir(10)', 'Rvir(11)']

    # Train (if not already trained)
    if not osp.exists(f'{taskdir}/model_justfinalhalos.pt'):

        # Create simple dataset (if not already created)
        if not os.path.exists(f'{datapath}{dataset}_finalhalosdata.pkl'):
            usetarg_idxs = [all_labelnames.index(targ) for targ in all_labelnames] 
            used_featidxs = [all_featnames.index(f) for f in all_featnames if f in use_feats]
            allgraphs = pickle.load(open(f"{datapath}{dataset}.pkl", 'rb'))
            traintransformer_path = osp.expanduser(f"{datapath}{dataset}_QuantileTransformer_train.pkl") 
            ft_train = pickle.load(open(traintransformer_path, 'rb')) # SHOULD REALLY BE USING SEPERATE TRANSFORMERS FOR TRAIN AND TEST BUT ACCIDENTALLY DIDNT SAVE TEST ONE YET
            sm_traindata, sm_testdata = data_utils.data_finalhalos(allgraphs, used_featidxs, usetarg_idxs, ft_train) # data_utils.data_fake()
            pickle.dump((sm_traindata, sm_testdata), open(f'{datapath}{dataset}_finalhalosdata.pkl', 'wb'))
        
        # Train simple model
        sm_traindata, sm_testdata = pickle.load(open(f'{datapath}{dataset}_finalhalosdata.pkl', 'rb'))
        sm_trainloader = DataLoader_notgeom(sm_traindata, batch_size=256, shuffle=True, num_workers=2)
        sm_testloader = DataLoader_notgeom(sm_testdata, batch_size=256, shuffle=True, num_workers=2)
        in_channels = len(next(iter(sm_traindata))[0])
        out_channels = len(next(iter(sm_traindata))[1])
        sm_model = models.SimpleNet(100, in_channels, out_channels)
        simple_train(sm_model, sm_trainloader, n_epochs = 100, loss_fn = nn.MSELoss())
        torch.save(sm_model.state_dict(), f'{taskdir}/model_justfinalhalos.pt')

    # Test
    sm_model = models.SimpleNet(100, in_channels, out_channels)
    sm_model.load_state_dict(torch.load(f'{taskdir}/model_justfinalhalos.pt'))
    sm_testys, sm_testpreds = simple_test(sm_model, sm_testloader)
    sm_metrics = {'sigma': np.std(sm_testpreds - sm_testys, axis=0),
                'R': np.array([stats.pearsonr(sm_testys[:,i], sm_testpreds[:,i]).statistic for i in range(sm_testys.shape[1])])}
    pickle.dump((sm_testys, sm_testpreds, sm_metrics), open(f'{taskdir}/model_justfinalhalos_testres.pkl', 'wb'))

# Train simple model
def simple_train(model, trainloader, n_epochs, loss_fn):
  
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  model.train()
  for e in range(n_epochs):
    tot_loss = 0
    for x, y in trainloader:
      optimizer.zero_grad()
      yhat = model(x)
      loss = loss_fn(yhat, y)
      loss.backward()
      optimizer.step()
      tot_loss += loss.item()

    print(f'[Epoch {e + 1}] avg loss: {np.round(tot_loss/len(trainloader),5)}')

# Test simple model
def simple_test(model, testloader):

    model.eval()
    with torch.no_grad():
        preds = []
        ys = []
        for x, y in testloader:
            yhat = model(x)
            preds.append(yhat)
            ys.append(y)
        preds = torch.cat(preds, dim=0)
        ys = torch.cat(ys, dim=0)
        preds = preds.detach().numpy()
        ys = ys.detach().numpy()
    
    return ys, preds

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
    models_list = [exp for exp in os.listdir(taskdir) if os.path.isdir(os.path.join(taskdir, exp)) and exp.startswith('e') and exp not in ['exp_tiny', 'exp1']]
    cols = ['exp', 'DS', 'avg val sig', 'avg test sig', 'base', 'loss func', 'n epochs', 'stop epoch', 'feats', 'targs']
    models = pd.DataFrame(columns = cols)

    # Add final halo only model for comparison
    _, _, sm_test_results = pickle.load(open(osp.join(taskdir, 'model_justfinalhalos_testres.pkl'), 'rb'))
    info_sm = ['MLP', 'vol100, z0', '',
               np.round(np.mean(sm_test_results['sigma']),2), # mean over all targets
               '', '', '', '',
               '[#scale(0), desc_scale(2), Mvir(10), Rvir(11)]', '[U, B, V, K, g, r, i, z]']
    models.loc[0] = info_sm

    # Add all other models
    for model in models_list:
        try: config = json.load(open(osp.join(taskdir, model, 'expfile.json'), 'rb'))
        except FileNotFoundError:
            print(f"ignoreing {model} (has no expfile.json)")
            continue
        try: results = CPU_Unpickler(open(osp.join(taskdir, model, 'result_dict.pkl'), 'rb')).load()
        except FileNotFoundError:
            print(f"Ignoreing {model} (has no result_dict.pkl)")
            continue
        try: 
            _, _, test_results = pickle.load(open(osp.join(taskdir, model, 'testres.pkl'), 'rb'))
            test_sig = np.round(np.mean(test_results['sigma']), 2) # mean over all targets
        except FileNotFoundError:
            test_sig = ''
        vol = config['data_params']['data_path'].replace('~/ceph/Data/','')[:-1]
        ds = config['data_params']['data_file'].replace('.pkl','')
        info = [model,
                f'{vol}, {ds}',
                np.round(np.mean(results['val sig final test']),2), # mean over all targets
                test_sig,
                config['run_params']['model'], 
                config['run_params']['loss_func'], 
                config['run_params']['n_epochs'], 
                results['epochexit'], 
                config['data_params']['use_feats'], 
                config['data_params']['use_targs']]
        models.loc[len(models)] = info

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