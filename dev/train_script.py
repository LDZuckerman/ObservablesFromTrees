import torch, pickle, time, os
import numpy as np
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from importlib import __import__
import random
import string
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
sys.path.insert(0, 'ObservablesFromTrees/dev/')
from dev import run_utils
import json



def run(run_params, data_params, learn_params, hyper_params, out_pointer, run_id, gpu): # "train_model"
    """
    For each of n_trials, train and validate (or train and test) model
    """
    
    ############### SET UP ################
    
    # Set up epoch counting 
    n_epochs = run_params['n_epochs'] # total num epochs
    val_epoch = run_params['val_epoch']
    exitepochs = [] # epoch at which exited training (or list of epochs if n_trials > 1), if early stopping 
    
    # Get loss function 
    if run_params['seed']: torch.manual_seed(42)
    loss_func = get_loss_func(run_params['loss_func'])
    
    # Set learning rate and schedular
    lr_init = learn_params['learning_rate'] # If no warmup (e.g. if schedule = onecycle, initial learning rate should be max learning rate)
    if learn_params['warmup'] and learn_params["schedule"] in ["warmup_exp", "warmup_expcos"]:
        lr_init = lr_init/(learn_params['g_up'])**(learn_params['warmup']) # If warmup, adjust initial learning rate
    lr_scheduler = get_lr_schedule(schedule  = learn_params['schedule']) 

    # Load data and add num in/out channels to hyper_params
    targ_names = data_params['use_targs'] 
    train_data, val_data = load_data(data_params) # REALLY TEST_DATA IF TEST=1 IN JSON!!!
    n_targ = len(train_data[0].y)
    if len(train_data[0].y) != len(targ_names): raise ValueError(f"Number of targets in data ({len(train_data[0].y)}) does not match number of targets in data_params ({len(targ_names)})")
    if train_data[0].x.numpy().shape[1] != len(data_params['use_feats']): raise ValueError(f"Number of features in data ({train_data[0].x.numpy().shape[1]}) does not match number of features in data_params ({len(data_params['use_feats'])})")
    val_loader = DataLoader(val_data, batch_size=run_params['batch_size'], shuffle=0, num_workers=run_params['num_workers']) # REALLY TEST_LOADER IF TEST=1 IN JSON!   ##never shuffle test
    hyper_params['in_channels']= train_data[0].x.numpy().shape[1] # construct_dict['hyper_params']['in_channels']=n_feat
    hyper_params['out_channels']= n_targ # construct_dict['hyper_params']['out_channels']=n_targ

    # Initialize evaluation metrics 
    # metric = get_metrics(run_params['metrics']) # FOR NOW JUST USE SCATTER SIGMA 
    ###
    # BUT WHAT IS ACC VS SIGMA THEN? CHRISTIAN WAS STD AS METRIC, BUT THEN WAS ALSO ADDING METRIC TO ACCS
    #  When metric is test_mutli_varrho, the thing it returns is std(ys-preds).
    #  He appends this to acc, but also prints it as scatter.
    #  And then after the final testing on val, the scatter he adds to the result dict is the same thing (std(ys-preds))
    #  So why is he storing both accs and metrics?
    #  And why is he calling it acc when YOU CANT REALLY DEFINE ACCURACY FOR REGREESION, and anyway, acc is generally higher = better, but scatter is lower = better (more like an error metric)?
    #  And why use std of errors as metric to define best model? why not use some actual error metric like rmse?
    #  Maybe ill just change everything called acc to metric as and it will be sigma (scatter), and then save rmse as rmse. Nothing called acc to aviod confusion. 
    ###
    performance_plot = get_performance_plot(run_params['performance_plot'])
    # tr_accs_alltrials, va_accs_alltrials, scatter_alltrials, = [], [], [] # lists in case n_trials > 1  # REALLY TEST_ACCS IF TEST IF TEST=1 IN JSON
    # preds_alltrials, lowest_alltrials = [], [] # lists in case n_trials > 1 
    
    ############### LOOP THROUGH TRIALS ###################
    
    # Perform experiment (either once or n_trials times)
    n_trials = run_params['n_trials']
    log = run_params['log']
    for trial in range(n_trials):

        # Set logging and state-saving dirs
        if n_trials>1:
            run_name_n = run_id+f'_{trial+1}_{n_trials}'
            out_dir_n = osp.join(out_pointer, run_name_n)
        else:
            run_name_n = run_id 
            out_dir_n = out_pointer
        if log:
            log_dir_n = osp.join(out_pointer+'/TB_logs/', run_name_n) # for TensorBoard events files
            from torch.utils.tensorboard import SummaryWriter 
            writer=SummaryWriter(log_dir=log_dir_n)
        
        # Initialize model, dataloaders, scheduler, and optimizer
        model = setup_model(run_params['model'], hyper_params) # print(f"N_params: {sum(p.numel() for p in model.parameters())}", flush=True)
        train_loader = DataLoader(train_data, batch_size=run_params['batch_size'], shuffle=True, num_workers=run_params['num_workers']) #shuffle training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        scheduler = lr_scheduler(optimizer, **learn_params, total_steps=n_epochs*len(train_loader))
       
        # If gpu, feed to accelerator
        if gpu == '1':
            from accelerate import Accelerator 
            accelerator = Accelerator() 
            device = accelerator.device
            _, _, val_loader = accelerator.prepare(model, optimizer, val_loader) # REALLY TEST_LOADER IF TEST=1 IN JSON!!!
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else: accelerator = None
        
        # Initialize accuracy and error tracking
        lowest_metric = np.array([np.inf]*n_targ) 
        low_ys = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        low_pred = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        tr_losses = [] # add this to track training loss for each epoch, instead of just accuracy when tested on training set every n_epochs epochs
        tr_metrics, va_metrics = [],[] # acc is really 'metric' which is sigma (scatter) # REALLY TE_ACC IF TEST=1 IN JSON!!!
        tr_rmses, va_rmses = [],[]
        lrs = [] # add this to track current lr for all val epochs

        # Set up epoch counting 
        epochexit = None
        counter = 0  # epochs since last validation improvement (was "early_stop")
        start = time.time()
        k = 0
        
        ###################### TRAIN (n epochs) #########################
        
        # Perform training
        s = '' if n_trials==1 else f' FOR TRIAL {trial+1} of {n_trials}'
        print(f'\nBEGINING TRAINING{s} ({n_epochs} epochs total)\n', flush=True)
        for epoch in tqdm(range(n_epochs)):

            # Train for one epoch (NOTE: trainloss is the total loss that includes the other losses returned)
            trainloss, err_loss, sig_loss, rho_loss, l1_loss, l2_loss = train(epoch, learn_params["schedule"], model, train_loader, run_params, gpu, n_targ, loss_func, accelerator, optimizer, scheduler)
            tr_losses.append(trainloss) # one value (not one for each target)
            
            # Step learning rate scheduler
            if learn_params["schedule"]!="onecycle": # onecycle steps per batch, not epoch
                scheduler.step(epoch)
                
            ################# EVALUATE ON VAL (if val epoch) ######################
            
            # If this is a val epoch, perform validation
            if (epoch+1)%val_epoch == 0: 
                
                # Perform validation   #train_metric, val_metric, lowest_metric, low_ys, low_pred, k, early_stop = get_val_results(model, run_params, data_params, train_loader, val_loader, optimizer, epoch, metric, k, lowest_metric, save, val_epoch, early_stop, n_targ, low_ys, low_pred, out_dir_n)
                tr_ys, tr_pred = test(train_loader, model, n_targ, run_params['loss_func'], get_varrho=False)
                va_ys, va_pred = test(val_loader, model, n_targ, run_params['loss_func'], get_varrho=False)
                train_metric = np.std(tr_pred - tr_ys, axis=0) # sigma for each targ  # metric(tr_ys, tr_pred)
                val_metric = np.std(va_pred - va_ys, axis=0) # sigma for each targ  # metric(va_ys, va_pred)  # REALLY TEST_METRIC IF TEST=1 IN JSON!!!
                train_rmse =  np.sqrt(np.mean((tr_ys-tr_pred)**2, axis=0)) # RMSE for each targ
                val_rmse =  np.sqrt(np.mean((va_ys-va_pred)**2, axis=0)) # RMSE for each targ 
                if k == 0: # If this is first val epoch
                    lowest_metric = val_metric
                    low_ys = va_ys
                    low_pred = va_pred
                    k += 1
                if np.any(val_metric < lowest_metric): # If this is the current best model (accuracy is better than previous best for at least one target)
                    mask = val_metric < lowest_metric
                    index = np.arange(n_targ)[mask]
                    lowest_metric[mask] = val_metric[mask]
                    low_ys[:,index] = va_ys[:,index]
                    low_pred[:,index] = va_pred[:,index]
                    counter = 0 # reset time since last validation improvement to 0
                    torch.save(model.state_dict(), osp.join(out_dir_n,'model_best.pt'))
                else:
                    counter += val_epoch # add num epochs we've run to counter
                                 
                # Add metrics for this epoch  
                tr_metrics.append(train_metric) 
                va_metrics.append(val_metric)  # REALLY TEST_ACCS IF TEST=1 IN JSON!!!
                tr_rmses.append(train_rmse)
                va_rmses.append(val_rmse)
                lr_curr = optimizer.state_dict()['param_groups'][0]['lr']
                lrs.append(lr_curr)
                last10val = np.median(va_metrics[-(10//val_epoch):], axis=0)  # REALLY LAST10TEST IF TEST=1 IN JSON!!!
                
                # Print results and add results to logger
                print_results_and_log(run_params, log, writer, epoch, lr_curr, trainloss, err_loss, sig_loss, rho_loss, l1_loss, l2_loss, train_metric, val_metric, lowest_metric, last10val, counter, n_targ, targ_names)
                
                # Every 5th val epoch make (or update - will overwrite each time?) plots 
                if (epoch+1)%(int(val_epoch*5))==0 and log:
                    figs = performance_plot(va_ys, va_pred, targ_names)
                    for fig, label in zip(figs, targ_names):
                        writer.add_figure(tag=f'{run_name_n}_{label}', figure=fig, global_step=epoch+1)
                
            #################  CHECK WHETHER TO STOP EARLY #################
                        
            if run_params['early_stopping']:
                if counter > run_params['patience']:
                    print(f'Exited after {epoch+1} epochs due to early stopping', flush=True)
                    epochexit = epoch
                    break
            
        stop=time.time()
        spent=stop-start
                                       
        ################ ADD DATA FROM THIS TRIAL TO LISTS ########################
        
        if epochexit == None: # if never had to exit early (if count was never > patience, or we weren't doing early stopping)
            epochexit = n_epochs
        exitepochs.append(epochexit)
        # va_accs_alltrials.append(va_acc) # REALLY TEST_ACCS IF TEST=1 IN JSON!!!
        # tr_accs_alltrials.append(tr_acc)
        # lowest_alltrials.append(lowest_metric)

        print(f"\nTRAINING COMPLETE{s} (total {spent:.2f} sec, {spent/epochexit:.3f} sec/epoch, {len(train_loader.dataset)*epochexit/spent:.0f} trees/sec)\n", flush=True)
          
        ###################### "TEST" ON VAL AND, RESULTS TO LISTS, PLOT #########################
        
        # Perform testing -> doing it again instead of using last val metrics just becasue perhaps last few training epochs werent done before last val epoch (depends on val_epoch and n_epochs)
        ys, pred, vars, rhos = test(val_loader, model, n_targ, run_params['loss_func'], get_varrho=True)
        if run_params['save_plots']:
            # WHY DID HE HAVE THIS TWICE ys, pred, xs, Mh, vars, rhos = test(val_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"]) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
            figs = performance_plot(ys, pred, targ_names)
            for fig, label in zip(figs, targ_names): # label_names[data_params["targets"]]
                fig.savefig(f'{out_dir_n}/performance_ne{n_epochs}_{label}.png')
        final_testonval_preds = pred
        final_testonval_ys = ys
        final_testonval_metric = np.std(ys-pred, axis=0)
        final_testonval_rmse = np.sqrt(np.mean((ys-pred)**2, axis=0))
        # preds_alltrials.append(pred)
        print(f"\nFINAL TESTING RMSE AND SCATTER{s}: {np.round(final_testonval_rmse, 4)}, {np.round(final_testonval_metric, 4)}", flush=True)
        
        ###################### SAVE PARAMS AND RESULTS (for this trial) #########################
  
        # Make params dict to add to writer
        paramsf = dict(list(data_params.items()) + list(run_params.items()) + list(hyper_params.items()) + list(learn_params.items()))
        paramsf["targets"] = str(data_params['use_targs']) # int("".join([str(i+1) for i in data_params["use_targs"]]))
        paramsf["use_feats"] = str(data_params["use_feats"])
        N_p=sum(p.numel() for p in model.parameters())
        N_t=sum(p.numel() for p in model.parameters() if p.requires_grad)
        paramsf['N_params']=N_p
        paramsf['N_trainable']=N_t
        from six import string_types
        for k, v in paramsf.items():  # before adding hparams to write, must change any lists to string
            if isinstance(v, list):
                v_new = '['+''.join(f'{x} ' for x in v).rstrip()+']'
                paramsf[k] = v_new
            if (isinstance(v, int) or isinstance(v, float) or isinstance(v, string_types) or isinstance(v, bool) or isinstance(v, torch.Tensor) or isinstance(v, list)) == False: 
                raise ValueError(f'paramsf({k}) is {v} ({type(v)}) -> must be int, float, string, bool, Tensor, or list')  

        # Make metrics dict to add to writer
        last20 = np.median(va_metrics[-(20//val_epoch):], axis=0) # REALLY TE_ACC IF TEST=1 IN JSON!!!
        last10 = np.median(va_metrics[-(10//val_epoch):], axis=0)
        metricf = {'epoch_exit':epoch}
        for i in range(n_targ):
            metricf[f'scatter_{targ_names[i]}'] = final_testonval_metric[i]
            metricf[f'lowest_{targ_names[i]}'] = lowest_metric[i]
            metricf[f'last20_{targ_names[i]}'] = last20[i]
            metricf[f'last10_{targ_names[i]}'] = last10[i]
        
        # Add to params and metrics dicts to writer and save results dict as pickle
        writer.add_hparams(paramsf, metricf, run_name=run_name_n) 
        result_dict={'train loss each epoch': tr_losses,            # [n_epochs]
                     'lr each val epoch': lrs,                      # [n_epochs/val_epoch]    
                     'val scatter each val epoch': va_metrics,      # [n_epochs/val_epoch, n_targs]
                     'train scatter each val epoch': tr_metrics,    # [n_epochs/val_epoch, n_targs]
                     'val rmse each val epoch': va_rmses,           # [n_epochs/val_epoch, n_targs]
                     'train rmse each val epoch': tr_rmses,         # [n_epochs/val_epoch, n_targs]
                     'val sig final test': final_testonval_metric,  # [n_targs]
                     'val rmse final test': final_testonval_rmse,   # [n_targs]
                     'val preds final test': final_testonval_preds, # [n_val_samples, n_targs]
                     'val ys final test': final_testonval_ys,       # [n_val_samples, n_targs]
                      #'ys': ys, 'pred': pred, 'low_ys': low_ys, 'low_pred': low_pred, 'low':lowest_metrics, 'vars': vars, 'rhos': rhos,  # Why is it usefull to save these at all? 
                     'epochexit': epochexit}
        with open(f'{out_dir_n}/result_dict.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle)
        # with open(f'{out_dir_n}/construct_dict.pkl', 'wb') as handle: # Just save exp file instead. Only thing that getting updated during run was adding in/out feats in construct_dict['hyper_params'], but that should be clear from use_targs and use_feats (and above I'm checking that they are the same)
        #     pickle.dump(construct_dict, handle)
        
    print(f'\nFINISHED ALL TRIALS', flush=True)


def load_data(data_params): 
    '''
    Load data from pickle, transform, remove unwanted features and targets, and split into train and val or train and test
    NOTE: For graph data like this need to apply fitted transformers to each graph individually (cant just stack like for images), but want to fit to train and val seperately
          Cant think of a better way than looping through data twice (once to fit transformer, once to transform)
          Chrisitan just used the same transfomer on train and test i think (probabaly ok if that transformer was just fitted on train or trainval?)
    '''

    # Load data from pickle
    data_path = data_params['data_path']
    data_file = data_params['data_file']
    print(f'Loading data from {data_path}{data_file}', flush=True)
    data = pickle.load(open(osp.expanduser(f'{data_path}/{data_file}'), 'rb'))

    # Split into train and val or train and test
    splits = data_params['splits']
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

    # Use dataset meta file to see names of all feats and targs in dataset
    data_meta_file = osp.expanduser(osp.join(data_path, data_file.replace('.pkl', '_meta.json')))
    data_meta = json.load(open(data_meta_file, 'rb'))
    all_label_names = data_meta[0]['obs meta']['label_names'] 
    all_featnames = data_meta[1]['tree meta']['featnames']
                
    # Fit transformers to all trainval graph data and all test graph data. Note: Use all cols for transform 
    transfname = data_params['transf_name']
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
            print(f'   Fitting transformer to test data (will save in {traintransformer_path})', flush=True)
            ft_test = run_utils.fit_transformer(trainval_data, all_featnames, save_path=testtransformer_path, transfname=data_params['transf_name']) 
        else: 
            print(f'   Applying transformer to test data (loaded from {traintransformer_path})', flush=True)
            ft_test = pickle.load(open(testtransformer_path, 'rb')) 

    # Transform each graph, and remove unwanted features and targets. 
    print(f"   Selecting desired features {data_params['use_feats']} and targets {data_params['use_targs']}", flush=True)
    usetarg_idxs = [all_label_names.index(targ) for targ in data_params['use_targs']] # targets = data_params['targets']
    use_featidxs = [all_featnames.index(f) for f in all_featnames if f in data_params['use_feats']] #  # have use_feats be names not idxs! in case we want to train on fewer features without recreating dataset
    trainval_out = []
    for graph in trainval_data:
        x = graph.x
        if transfname != "None": 
            x = ft_train.transform(graph.x)
        x = torch.tensor(x[:, use_featidxs])
        trainval_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y[usetarg_idxs]))
    test_out = []
    for graph in test_data:
        x = graph.x
        if transfname != "None": 
            x = ft_test.transform(x)
        x = torch.tensor(x[:, use_featidxs])
        test_out.append(Data(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y[usetarg_idxs]))

    # If "test" run return train+val and test (NOTE: Christian uses 'test' to mean training the same model again completely from scratch on train+val, then testing on test)
    if data_params['test']:

        return trainval_out, test_out # merged train+val, test
   
   # If "train" run, fit or load transformer and split into train and val
    else:
        val_pct = splits[1]/(splits[0]+splits[1]) # pct of train
        train_out = trainval_out[:int(len(trainval_out)*(1-val_pct))]
        val_out = trainval_out[int(len(trainval_out)*(val_pct)):]
        
        return train_out, val_out 
    

def train(epoch, schedule, model, train_loader, run_params, gpu, n_targ, loss_func, accelerator, optimizer, scheduler):
    '''
    Train for one epoch
    '''
    
    model.train()

    l1_lambda = run_params['l1_lambda']
    l2_lambda = run_params['l2_lambda']

    if gpu == '1': init = torch.cuda.FloatTensor([0])
    else: init = torch.FloatTensor([0])
    er_loss = torch.clone(init)
    si_loss = torch.clone(init)
    rh_loss = torch.clone(init)

    return_loss = 0
    for data in train_loader: 
        if gpu == '0':
            data = data.to('cpu')
        if run_params["loss_func"] in ["L1", "L2", "SmoothL1"]: 
            out = model(data)  
            loss = loss_func(out, data.y.view(-1,n_targ))
        if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "GaussNd"]:
            out, var = model(data)  
            loss, err_loss, sig_loss = loss_func(out, data.y.view(-1,n_targ), var) # loss = total loss = err_los + sig_loss
            er_loss+=err_loss
            si_loss+=sig_loss
        if run_params["loss_func"] in ["Gauss2d_corr"]:
            out, var, rho = model(data)  
            loss, err_loss, sig_loss, rho_loss = loss_func(out, data.y.view(-1,n_targ), var, rho)
            er_loss+=err_loss
            si_loss+=sig_loss
            rh_loss+=rho_loss
        if run_params["loss_func"] in ["Gauss4d_corr"]:
            out, var, rho = model(data)  
            loss, err_loss, sig_loss = loss_func(out, data.y.view(-1,n_targ), var, rho)
            er_loss+=err_loss
            si_loss+=sig_loss
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) 
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm # add other terms to loss
        return_loss += loss
        if gpu == '1':
            accelerator.backward(loss)
        else: 
            loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
        if schedule == "onecycle":
            scheduler.step(epoch)
    # if epoch==0:  writer.add_graph(model,[data])   #Doesn't work right now but could be fun to add back in

    return return_loss, er_loss, si_loss, rh_loss, l1_lambda * l1_norm, l2_lambda * l2_norm           


def print_results_and_log(run_params, log, writer, epoch, lr_curr, trainloss, err_loss, sig_loss, rho_loss, l1_loss, l2_loss, train_metric, val_metric, lowest_metric, last10val, counter, n_targ, targ_names):
    
    # Add val results to TensorBoard logger
    if log:
        writer.add_scalar('train_loss', trainloss,global_step=epoch+1)
        writer.add_scalar('learning_rate', lr_curr, global_step=epoch+1)
        for i in range(n_targ):
            writer.add_scalar(f'last10val_{targ_names[i]}', last10val[i], global_step=epoch+1) # WAS CALLING LAST10VAL LAST10TEST BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
            writer.add_scalar(f'train_scatter_{targ_names[i]}', train_metric[i], global_step=epoch+1)
            writer.add_scalar(f'val_scatter_{targ_names[i]}', val_metric[i], global_step=epoch+1) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
            writer.add_scalar(f'best_scatter_{targ_names[i]}', lowest_metric[i], global_step=epoch+1)
    
    # Print train and val results 
    print(f'\nEpoch {int(epoch+1)} (learning rate {lr_curr:.2E})', flush=True)
    print(f'\tTrain losses: total = {trainloss.cpu().detach().numpy():.2E},  L1 regularization = {l1_loss.cpu().detach().numpy():.2E},  L2 regularization = {l2_loss.cpu().detach().numpy():.2E}', flush=True)
    print(f'\tTrain metric (scatter): {np.round(train_metric,4)}', flush=True)
    if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "Gauss2d_corr", "Gauss4d_corr", "Gauss_Nd"]:
        print(f'\t[Err/Sig/Rho]: {err_loss.cpu().detach().numpy()[0]:.2E}, {sig_loss.cpu().detach().numpy()[0]:.2E}, {rho_loss.cpu().detach().numpy()[0]:.2E}', flush=True)       
    print(f'\tVal metric (scatter): {np.round(val_metric,4)}, lowest was {np.round(lowest_metric,4)}', flush=True)  # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
    print(f'\tMedian metric (scatter) for last 10 val epochs: {np.round(last10val,4)} ({counter} epochs since last improvement)', flush=True)
    
    return
  

def test(loader, model, n_targ, l_func, get_varrho = False):
    '''
    Returns targets and predictions
    '''
    ys, pred, vars, rhos = [],[], [], []
    model.eval()
    with torch.no_grad():
        for data in loader: 
            rho = torch.IntTensor(0)
            var = torch.IntTensor(0)
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr", "Gauss4d_corr"]:
                out, var, rho = model(data) 
            ys.append(data.y.view(-1,n_targ))
            pred.append(out)
            vars.append(var)
            rhos.append(rho)
    ys = torch.vstack(ys)
    pred = torch.vstack(pred)
    vars = torch.vstack(vars)
    rhos = torch.vstack(rhos)

    if get_varrho:
        return ys.cpu().numpy(), pred.cpu().numpy(), vars, rhos   
    else:
        return ys.cpu().numpy(), pred.cpu().numpy()
    

def get_metrics(metric_name):
    '''
    Returns functions to scrutinize performance on the fly
    '''

    import dev.metrics as metrics
    metrics=getattr(metrics, metric_name)
    return metrics


def get_loss_func(name):
    '''
    Load loss function from loss_funcs file
    '''

    import dev.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    try:
        l=loss_func()
    except:
        l=loss_func

    return l

def get_performance_plot(name):
    '''
    Load plotting function from plot_utils file
    '''

    import dev.plot_utils as evals
    performance_plot = getattr(evals, name)

    return performance_plot 

def setup_model(model_name, hyper_params):
    '''
    Load model object from model folder
    '''

    import dev.models as models
    model = getattr(models, model_name)
    model = model(**hyper_params)

    return model

def get_lr_schedule(schedule):
    '''
    Load schedular object from lr_schedule file
    '''

    import dev.lr_schedule as lr_module
    schedule_class = getattr(lr_module, schedule)

    return schedule_class

# def get_val_results(model, run_params, data_params, train_loader, test_loader, optimizer, epoch, metric, k, lowest_metric, val_epoch, early_stop, n_targ, low_ys, low_pred, out_dir_n):
   
#     if run_params['metrics']!='test_multi_varrho':
#         train_metric, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
#         test_metric, ys, pred = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
#     else:
#         train_metric, _, _, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
#         test_metric, ys, pred, vars, rhos = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
#     if k == 0: # If this is first val epoch
#         lowest_metric = test_metric
#         low_ys = ys
#         low_pred = pred
#         k+=1
#     if np.any(test_metric < lowest_metric): # If this is the current best model
#         mask = test_metric < lowest_metric
#         index = np.arange(n_targ)[mask]
#         lowest_metric[mask] = test_metric[mask]
#         low_ys[:,index] = ys[:,index]
#         low_pred[:,index] = pred[:,index]
#         early_stop = 0
#         if run_params['save']:
#             torch.save(model.state_dict(), osp.join(out_dir_n,'model_best.pt'))
#     else:
#         early_stop+=val_epoch

#     return train_metric, test_metric, lowest_metric, low_ys, low_pred, k, early_stop