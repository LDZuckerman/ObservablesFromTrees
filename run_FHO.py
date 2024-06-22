import pickle, json
import numpy as np
import pandas as pd
import torch
import os, sys
import os.path as osp
from torch.utils.data import DataLoader as DataLoader_notgeom
from torch import nn
import torch
from torch import nn
import scipy.stats as stats
import sklearn.metrics as skmetrics
import copy
from accelerate import Accelerator 
try: 
    from dev import data_utils, run_utils, models
except ModuleNotFoundError:
    try:
        sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
        from ObservablesFromTrees.dev import data_utils, run_utils, models
    except ModuleNotFoundError:
        sys.path.insert(0, 'ObservablesFromTrees/dev/')
        from dev import data_utils, run_utils, models


def run_simplemodel(taskdir, d, retrain=False, gpu=False):

    name = d['exp_id']
    print(f"Running model {name}\nConfig dict {d}")

    # Save config for future reference
    json.dump(d, open(f'{taskdir}/FHO/{name}_config.json', 'w')) # f'../{taskdir}/FHO/{name}_config.json'

    # Get data
    dataset_path = d['data_params']['datapath']
    print(f"  Loading train and test data from {dataset_path}")
    traindata, valdata, testdata = pickle.load(open(f'{dataset_path}', 'rb')) 
    trainloader = DataLoader_notgeom(traindata, batch_size=256, shuffle=True)#, num_workers=2)
    valloader = DataLoader_notgeom(valdata, batch_size=256, shuffle=False)
    testloader = DataLoader_notgeom(testdata, batch_size=256, shuffle=False)
    in_channels = len(next(iter(traindata))[0])
    out_channels = len(next(iter(traindata))[1])

    # Delete current if 'retrain' = True
    modelpath =  f'{taskdir}/FHO/{name}.pt'
    if retrain:
        files = [f for f in os.listdir(f'{taskdir}/FHO/') if f.startswith(name)]
        print(f'  Retraining model {name} - deleting existing {files}')
        for f in files:
            os.remove(f'{taskdir}/FHO/{f}')

    # Train (if not already trained)
    predict_dist = d['run_params']['loss_func'] in ['GaussNd', 'Navarro', 'Navarro2', 'GaussNd_torch']
    # if predict_dist: model = models.SimpleDistNet(d['hyper_params'], in_channels, out_channels)
    # else: model = models.SimpleNet(100, in_channels, out_channels)
    if d['modelname'] == 'SimpleDistNet':
        model = models.SimpleDistNet(d['hyper_params'], in_channels, out_channels)
    elif d['modelname'] == 'SimpleDistNet2':
        model = models.SimpleDistNet2(d['hyper_params'], in_channels, out_channels)
    elif d['modelname'] == 'SimpleNet':
        model = models.SimpleNet(100, in_channels, out_channels)
    else:
        raise ValueError(f"Model name {d['modelname']} not recognized")
    if not os.path.exists(modelpath):
        train_results = simple_train(model, trainloader, d['run_params'], d['learn_params'], valloader, modelpath, gpu) # YES, SHOULDNT USE SAME TEST SET FOR VALIDATION, BUT IT REALLY SHOULDNT AFFECT ANYTHING AND THIS IS JUST FOR DEBUGGING
        print(f'  Training complete\nSaving trained model to {modelpath} and training results to {taskdir}/FHO/{name}_train_results.pkl')
        if not os.path.exists(f'{taskdir}/FHO/'): os.makedirs(f'{taskdir}/FHO/')
        torch.save(model.state_dict(), modelpath)
        pickle.dump(train_results, open(f'{taskdir}/FHO/{name}_train_results.pkl','wb'))
    else: print(f'  Trained model already exists for {name}. Loading it back in to compute test results')

    # Test and save results       
    model.load_state_dict(torch.load(modelpath))
    print(f'  Testing') 
    ys, preds = simple_test(testloader, model, predict_dist=predict_dist) 
    yhats_samp = data_utils.get_yhats(preds, sample_method='large_sample') # Sample for computing metrics
    yhats_flat, ys_flat = data_utils.flatten_sampled_yhats(yhats_samp, ys)
    metrics = {'rmse': np.std(yhats_flat - ys_flat, axis=0), 
               'rho': np.array([stats.pearsonr(ys_flat[:,i], yhats_flat[:,i]).statistic for i in range(ys_flat.shape[1])]), 
               'R2': np.array([skmetrics.r2_score(ys_flat[:,i], yhats_flat[:,i]) for i in range(ys_flat.shape[1])]), 
               'bias': np.mean(yhats_flat - ys_flat, axis=0), # mean of the residuals
               'scatter': np.std(yhats_flat - ys_flat, axis=0)} # SD of the residuals
    print(f'  Saving test results to {modelpath.replace(".pt", "")}_testres.pkl')
    pickle.dump((ys, preds, metrics), open(f'{modelpath.replace(".pt", "")}_testres.pkl', 'wb')) # Return un-sampled preds

    return ys, preds, metrics


# Train simple model
def simple_train(model, trainloader, run_params, learn_params, valloader, modelpath, gpu, val_epoch=20):
                #model, trainloader, d['run_params'], d['learn_params'], valloader, modelpath, gpu)

    # Set train params
    n_epochs = run_params['n_epochs']
    patience = run_params['patience']
    lr = learn_params['learning_rate']
    clip = learn_params['clip']
    loss_fn = run_params['loss_func']
    predict_mu_sig = True if loss_fn in ['GaussNd', 'Navarro', 'Navarro2', 'GaussNd_torch'] else False
    loss_func = run_utils.get_loss_func(loss_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # If gpu, feed to accelerator 
    if gpu:
        accelerator = Accelerator() 
        _, _, valloader = accelerator.prepare(model, optimizer, valloader) 
        model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

    # Train for n_epochs (or untill patience is reached)
    print(f'  Training ({n_epochs} epochs with {loss_fn} loss)')
    model.train()
    totloss_all, errloss_all, sigloss_all, val_rmse_all, val_preds_all = [], [], [], [], []
    counter = 0
    k = 0
    for e in range(n_epochs):

        # Loop over trainloader
        epoch_totloss, epoch_errloss, epoch_sigloss = 0, 0, 0
        for x, y in trainloader:
            totloss, errloss, sigloss = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            optimizer.zero_grad()
            if predict_mu_sig:
                muhat, logsighat = model(x)#, debug=True)
                sighat = torch.exp(logsighat)
                if loss_fn in ['GaussNd', 'Navarro','Navarro2']: 
                    totloss, errloss, sigloss = loss_func(muhat, y, sighat)
                elif loss_fn == 'GaussNd_torch': # Torch GaussianNLLLoss expects var, not sig, and doesnt seperate loss terms
                    totloss = loss_func(muhat, y, sighat**2)
            else: 
                yhat = model(x)
                totloss = loss_func(yhat, y)
            #print(type(totloss.item()), type(epoch_totloss))
            epoch_totloss += totloss.item()
            epoch_errloss += errloss.item() # zero if not custom loss
            epoch_sigloss += sigloss.item() # zero if not custom loss
            totloss.backward()
            if eval(clip): nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
        totloss_all.append(np.round(epoch_totloss/len(trainloader),5))
        errloss_all.append(np.round(epoch_errloss/len(trainloader),5))
        sigloss_all.append(np.round(epoch_sigloss/len(trainloader),5))

        # Perform validation test and asess early stopping 
        if e % val_epoch == 0:
            ys, preds = simple_test(valloader, model, predict_mu_sig)
            val_rmse = np.std(preds[0] - ys, axis=0)
            val_rmse_all.append(val_rmse)
            val_preds_all.append(preds)
            if k == 0: # If this is first val epoch
                best_avg_val_rmse = np.mean(val_rmse)
                k += 1
            if np.mean(val_rmse) < best_avg_val_rmse: # Check if current "best" model (avg acc over all targs is better than previous best). NOTE: previously was using criterion that metric was better for only at least one target
                best_avg_val_rmse = np.mean(val_rmse)
                counter = 0 # reset time since last validation improvement to 0
                torch.save(model.state_dict(), modelpath) # print(f'\tNew "best" model found; saving in {modelfile}', flush=True)
            else:
                counter += val_epoch # add num epochs we've run to counter

        if counter >= patience: 
            print(f'    Early stopping at epoch {e} due to no improvement in validation loss')
            break
        print(f'    Epoch {e}', end="\r") 

    epochexit = e if counter >= patience else n_epochs
    train_results = {'loss_all':totloss_all, 'errloss_all':errloss_all, 'sigloss_all':sigloss_all, 'val_rmse_all':val_rmse_all, 'val_preds_all':val_preds_all, 'epochexit':epochexit} 

    return train_results


# Test simple model
def simple_test(loader, model, predict_dist):
    '''
    Returns targets and predictions
    '''
  
    ys, yhats, sigs, xs, = [],[], [], []  # add option to return xs
    model.eval()
    with torch.no_grad():
        for x, y in loader: 
            ys.append(y)
            xs.append(x)
            if predict_dist:
                pred_mu, pred_logsig = model(x)
                yhats.append(pred_mu)
                sigs.append(torch.exp(pred_logsig))
            else:
                yhat = model(x)
                yhats.append(yhat)
    ys = torch.vstack(ys).cpu().numpy()
    yhats = torch.vstack(yhats).cpu().numpy()
    xs = torch.vstack(xs).cpu().numpy()

    if predict_dist:
        sigs = torch.vstack(sigs).cpu().numpy()
        preds = [yhats, sigs]
    else:
        preds = yhats

    return ys, preds


if __name__ == "__main__":

    # Set base dict
    taskdir = 'Task_phot'
    dataset = 'DS2'
    vol = 'vol100'
    use_feats = ['Mvir(10)','vmax(16)']
    if taskdir == 'Task_phot':
        targ_type = 'phot'
        use_targs = ["U", "B", "V", "K", "g", "r", "i", "z"] # all 
        dpath = f"/mnt/home/lzuckerman/ceph/Data/{vol}/"
        dataset_path = f"{dpath}/{dataset}_{targ_type}_FHOdata_{use_feats}.pkl"
    if taskdir == 'Task_props':
        targ_type = 'props'
        use_targs = ["SubhaloBHMass", "SubhaloGasMass", "SubhaloStelMass", "SubhaloSFR", "SubhaloVmax"] # from Task_props exp1
        dpath = f"/mnt/home/lzuckerman/ceph/Data/{vol}/"
        dataset_path = f"{dpath}/{dataset}_{targ_type}_FHOdata_{use_feats}.pkl"
    if taskdir == 'Task_samProps':
        targ_type = 'samProps'
        use_feats = [' mBH black hole mass [1.0E09 Msun](27)', ' mcold cold gas mass in disk [1.0E09 Msun](15)', ' Metal_cold metal mass in cold gas [Zsun*Msun] (20)', 'mstar stellar mass [1.0E09 Msun](8)', ' sfrave100myr SFR averaged over 100 Myr [Msun/yr](23)', ' sfr instantaneous SFR [Msun/yr](21)'] # from Task_samProps exp2
        dpath = f'/mnt/home/lzuckerman/ceph/Data/sam/'
        dataset_path = f"{dpath}/{dataset}_FHOdata_{use_feats}.pkl"
    basedict = {'run_params': {'loss_func':'GaussNd_torch', 'n_epochs':1000}, 
                "learn_params":{ "learning_rate": ''},
                'hyper_params': {'hidden_layers':'', 'hidden_channels':'' , 'activation':'', 'learning_rate':''},
                'data_params': {'dataset':dataset, 'datapath':dataset_path, 'vol':vol, 'data_file':f'{dataset}z0', 'use_feats':use_feats, 'use_targs':use_targs}}

    # Create list of sweep dicts for hyper param search
    sweepid = '1'
    sweepdict = {'hidden_layers':[1, 3], 'hidden_channels':[32, 64, 128], 'activation':['LeakyReLU', 'Sigmoid', 'Tanh'], 'learning_rate':[0.1, 0.01]}
    diffsets = run_utils.perms(sweepdict) 
    diffsets = pd.DataFrame(data=diffsets, columns=sweepdict.keys())
    all_paramdicts = []
    for i in range(len(diffsets)):
        dict = copy.deepcopy(basedict)
        diffset = diffsets.iloc[i]
        for group_key in ['hyper_params', 'learn_params']: # just sweeping hyper params and learn params for now
            for key in basedict[group_key].keys(): 
                if key in sweepdict.keys():
                    dict[group_key][key] = diffset[key]
        dict['exp_id'] = f"FHO{sweepid}{'{:03d}'.format(i)}" # e.g. expS0001 for first experiment in sweep0
        all_paramdicts.append(dict)
    if len(all_paramdicts) > 999: raise ValueError(f"Too many experiments in sweep. Need to add more digits to exp_id to accomodate.")

    # Create data if not already created
    if not os.path.exists(dataset_path):
        print(f'  Final halos only dataset for {targ_type} with features {use_feats} not already created for {dataset}, creating now')
        data_utils.data_finalhalos(dpath, dataset, targ_type, use_feats, use_targs, savepath=dataset_path) # data_utils.data_fake()
    print(f'  Loading data from {dataset_path}')

    # Hyper-params search
    print(f"RUNNING HYPER PARAM SEARCH OF {len(all_paramdicts)} FHO EXPERIMENTS\n", flush=True)
    for config in all_paramdicts:
        resfile = f"{taskdir}/FHO/{config['exp_id']}_testres.pkl"
        if not osp.exists(resfile):
            sm_ys, sm_preds, _ = run_simplemodel(taskdir, config, gpu=True)
