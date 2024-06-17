import torch, pickle, time
import numpy as np
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter 
from accelerate import Accelerator 
from importlib import __import__
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
try: 
    from dev import data_utils, run_utils
except ModuleNotFoundError:
    try:
        sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
        from ObservablesFromTrees.dev import data_utils, run_utils
    except ModuleNotFoundError:
        sys.path.insert(0, 'ObservablesFromTrees/dev/')
        from dev import data_utils, run_utils

def run(run_params, data_params, learn_params, hyper_params, out_pointer, exp_id, gpu, low_output=False, proc=None): # "train_model"
    """
    Train and validate model (for one or multiple trials) and save results
    """

    # for k in ["hidden_channels", "conv_layers", "decode_layers"]: # WHY ARE THINGS GETTING SET AS A STRING WHEN RUNNING SWEEP??)
    #     if isinstance(hyper_params[k], str): hyper_params[k] = int(hyper_params[k]) 
    # for k in ["n_epochs", "n_trials", "batch_size", "val_epoch", "patience", "l1_lambda", "l2_lambda", "num_workers"]: # WHY ARE THINGS GETTING SET AS A STRING WHEN RUNNING SWEEP??)
    #     if isinstance(run_params[k], str): run_params[k] = int(run_params[k]) 
    # for k in learn_params.keys(): # WHY ARE THINGS GETTING SET AS A STRING WHEN RUNNING SWEEP??)
    #     if isinstance(learn_params[k], str) and k != "schedule": learn_params[k] = float(learn_params[k]) 
    if gpu not in [True, False]: raise ValueError(f"gpu must be True or False, not {gpu}")
    p = '' if proc == None else f'[proc {proc}] '
    
    ############### SET UP ################

    # If low output mode, overwrite output params
    if low_output:
        run_params['performance_plot'] = "None"
        run_params['log'] = "False"

    # Set up epoch counting 
    n_epochs = run_params['n_epochs'] # total num epochs
    val_epoch = run_params['val_epoch']
    exit_epoch = [] # epoch at which exited training (or list of epochs if n_trials > 1), if early stopping 
    
    # Get loss function
    if run_params['seed']: torch.manual_seed(42)
    loss_name = run_params['loss_func']
    loss_func = run_utils.get_loss_func(loss_name)
    hyper_params['get_sig'] = True if loss_name in ["GaussNd", "Navarro", "GaussNd_torch"] else False 
    hyper_params['get_cov'] = True if loss_name in ["GaussNd_corr"] else False

    # Set learning rate and schedular
    lr_init = learn_params['learning_rate'] # If no warmup (e.g. if schedule = onecycle, initial learning rate should be max learning rate)
    if learn_params['warmup'] and learn_params["schedule"] in ["warmup_exp", "warmup_expcos"]:
        lr_init = lr_init/(learn_params['g_up'])**(learn_params['warmup']) # If warmup, adjust initial learning rate
    lr_scheduler = get_lr_schedule(schedule  = learn_params['schedule']) 

    # Create subset dataset (dataset with feat/targ selections applied, transform applied, and split into train and test)
    subset_path = data_utils.get_subset_path(data_params, set = 'train')
    if not osp.exists(f'{subset_path}'):
        print(f"{p}Dataset subset {subset_path} does not yet exist", flush=True)
        train_data, val_data = data_utils.create_subset(data_params, return_set = 'train') 
    else: # for sweeps, should probabaly just load once and pass in.. but this doesnt really take that long
        print(f"{p}Dataset subset already exists. Loading from {subset_path}", flush=True)
        train_data, val_data = pickle.load(open(subset_path, 'rb'))

    # Load data and add num in/out channels to hyper_params
    targ_names = data_params['use_targs']
    n_targ = len(train_data[0].y)
    if len(train_data[0].y) != len(targ_names): raise ValueError(f"Number of targets in data ({len(train_data[0].y)}) does not match number of targets in data_params ({len(targ_names)})")   
    if train_data[0].x.numpy().shape[1] != len(data_params['use_feats']): raise ValueError(f"Number of features in data ({train_data[0].x.numpy().shape[1]}) does not match number of features in data_params ({len(data_params['use_feats'])})")   
    val_loader = DataLoader(val_data, batch_size=int(run_params['batch_size']), shuffle=0, num_workers=run_params['num_workers'])    
    hyper_params['in_channels']= train_data[0].x.numpy().shape[1] # construct_dict['hyper_params']['in_channels']=n_feat
    hyper_params['out_channels']= n_targ # construct_dict['hyper_params']['out_channels']=n_targ
    
    # Initialize evaluation metrics 
    print(eval(str(run_params['performance_plot'])) != None)
    if eval(str(run_params['performance_plot'])) != None:
        performance_plot = get_performance_plot(run_params['performance_plot']); print(f"{p}Line 84", flush=True) # returns None if performance_plot = None
        # tr_accs_alltrials, va_accs_alltrials, scatter_alltrials, preds_alltrials, lowest_alltrials = [], [], [], [], [] # lists in case n_trials > 1 

    ############### LOOP THROUGH TRIALS ###################
    
    # Perform experiment (either once or n_trials times)
    n_trials = run_params['n_trials']
    log = run_params['log']
    for trial in range(n_trials):

        # Set logging, state-saving, and result output dirs
        if n_trials>1:
            run_name_n = exp_id+f'_{trial+1}_{n_trials}'
            out_dir_n = osp.join(out_pointer, run_name_n)
            raise ValueError("Should probabaly change this.. I think I would prefer to have the trials all in the same folder, within the exp folder")
        else:
            run_name_n = exp_id 
            out_dir_n = out_pointer
        if eval(str(log)):
            log_dir_n = osp.join(out_pointer+'/TB_logs/', run_name_n) # for TensorBoard events files
            writer = SummaryWriter(log_dir=log_dir_n)
        else: writer = None
        if proc == None:
            resfile = f'{out_dir_n}/result_dict.pkl'
            modelfile = f'{out_dir_n}/model_best.pt'
        else: 
            resfile =  f'{out_dir_n}/{exp_id}_result_dict.pkl' # If this is part of a sweep, don't create individiual output folders, just result files
            modelfile = f'{out_dir_n}/{exp_id}_model_best.pt' # If this is part of a sweep, don't create individiual output folders, just result files
        # Initialize model, dataloaders, scheduler, and optimizer
        model = run_utils.get_model(run_params['model'], hyper_params) # print(f"N_params: {sum(p.numel() for p in model.parameters())}", flush=True)
        train_loader = DataLoader(train_data, batch_size=run_params['batch_size'], shuffle=True, num_workers=run_params['num_workers']) #shuffle training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        scheduler = lr_scheduler(optimizer, **learn_params, total_steps=n_epochs*len(train_loader))
       
        # If gpu, feed to accelerator 
        if gpu:
            accelerator = Accelerator() 
            device = accelerator.device
            _, _, val_loader = accelerator.prepare(model, optimizer, val_loader) 
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else: 
            accelerator = None
        if not next(model.parameters()).is_cuda:
            raise ValueError("Model is not on GPU. Is this intentional?")

        # Initialize accuracy and error tracking
        best_metric = np.array([np.inf]*n_targ) 
        # best_yhats = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        # best_ys = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        tot_losses, sig_losses, err_losses = [], [], [] # track training loss for each epoch
        tr_metrics, va_metrics = [],[] 
        tr_rmses, va_rmses = [],[]
        lrs = [] # track current lr for all val epochs

        # Set up epoch counting 
        epochexit = None
        counter = 0  # epochs since last validation improvement (was "early_stop")
        start = time.time()
        k = 0
        
        ###################### TRAIN (n epochs) #########################     
        # Perform training
        t0 = time.time()
        s = '' if n_trials==1 else f' FOR TRIAL {trial+1} of {n_trials}'
        print(f'{p}BEGINING TRAINING{s} OF {exp_id} ({n_epochs} epochs total)', flush=True)
        for epoch in tqdm(range(n_epochs)):

            # Train for one epoch (NOTE: trainloss is the total loss that includes the other losses returned)
            tot_loss, err_loss, sig_loss, l1_loss, l2_loss = train(epoch, learn_params["schedule"], model, train_loader, run_params, gpu, n_targ, loss_func, accelerator, optimizer, scheduler)
            tot_losses.append(tot_loss) 
            sig_losses.append(sig_loss) # this is sig_loss, unless predict_cov, then cov_loss
            err_losses.append(err_loss)

            # Step learning rate scheduler
            if learn_params["schedule"]!="onecycle": # onecycle steps per batch, not epoch
                scheduler.step(epoch)
                
            ################# EVALUATE ON VAL (if val epoch) ######################
            
            # If this is a val epoch, perform validation
            if (epoch+1)%val_epoch == 0: 
                
                # Perform validation   #train_metric, val_metric, lowest_metric, low_ys, low_pred, k, early_stop = get_val_results(model, run_params, data_params, train_loader, val_loader, optimizer, epoch, metric, k, lowest_metric, save, val_epoch, early_stop, n_targ, low_ys, low_pred, out_dir_n)
                predict_dist = True if hyper_params['get_sig'] or hyper_params['get_cov'] else False 
                tr_ys, tr_preds = run_utils.test(train_loader, model, n_targ, predict_dist)
                va_ys, va_preds = run_utils.test(val_loader, model, n_targ, predict_dist)
                if predict_dist: # If predicting mu and sig, test returns [pred_mu, pred_sig], and if predicting mu and cov, test returns [pred_mu, pred_cov]
                    tr_yhats = tr_preds[0]; va_yhats = va_preds[0]
                else:
                    tr_yhats = tr_preds; va_yhats = va_preds                
                train_metric = np.std(tr_yhats - tr_ys, axis=0) # sigma for each targ  # metric(tr_ys, tr_pred)
                val_metric = np.std(va_yhats - va_ys, axis=0) # sigma for each targ  # metric(va_ys, va_pred)  
                train_rmse =  np.sqrt(np.mean((tr_ys-tr_yhats)**2, axis=0)) # RMSE for each targ
                val_rmse =  np.sqrt(np.mean((va_ys-va_yhats)**2, axis=0)) # RMSE for each targ 
                if k == 0: # If this is first val epoch
                    best_metric = val_metric
                    best_preds = va_preds; # best_ys = va_ys
                    k += 1
                if np.mean(best_metric) < np.mean(val_metric): # Check if current "best" model (avg acc over all targs is better than previous best). NOTE: previously was using criterion that metric was better for only at least one target
                    # mask = val_metric < lowest_metric
                    # index = np.arange(n_targ)[mask]
                    # lowest_metric[mask] = val_metric[mask]
                    # low_ys[:,index] = va_ys[:,index]
                    # low_yhats[:,index] = va_yhats[:,index]
                    best_metric = val_metric
                    best_preds = va_preds; # best_ys = va_ys
                    counter = 0 # reset time since last validation improvement to 0
                    print(f'\tNew "best" model found; saving in {modelfile}', flush=True)
                    torch.save(model.state_dict(), modelfile)
                else:
                    counter += val_epoch # add num epochs we've run to counter
                                 
                # Add metrics for this epoch  
                tr_metrics.append(train_metric) 
                va_metrics.append(val_metric) 
                tr_rmses.append(train_rmse)
                va_rmses.append(val_rmse)
                lr_curr = optimizer.state_dict()['param_groups'][0]['lr']
                lrs.append(lr_curr)
                #last10val = np.median(va_metrics[-(10//val_epoch):], axis=0)  
                
                # Print results and add results to logger
                if not low_output:
                    print_results_and_log(eval(str(log)), loss_name, epoch, lr_curr, tot_loss, err_loss, sig_loss, l1_loss, l2_loss, train_metric, val_metric, best_metric, counter, n_targ, targ_names, writer)
                
                # Every 5th val epoch make plots (really update - will overwrite each time) 
                if eval(str(run_params['performance_plot'])) != None and eval(str(log)) and (epoch+1)%(int(val_epoch*5))==0:
                    figs = performance_plot(va_ys, va_preds, targ_names)
                    for fig, label in zip(figs, targ_names):
                        writer.add_figure(tag=f'{run_name_n}_{label}', figure=fig, global_step=epoch+1)
                
            #################  CHECK WHETHER TO STOP EARLY #################
                        
            if run_params['early_stopping']:
                if counter > run_params['patience']:
                    print(f'{p}Exited training of {exp_id} after {epoch+1} epochs due to early stopping', flush=True)
                    epochexit = epoch
                    break
            
        stop=time.time()
        spent=stop-start
                                       
        ################ ADD DATA FROM THIS TRIAL TO LISTS ########################
        
        if epochexit == None: # if never had to exit early (if count was never > patience, or we weren't doing early stopping)
            epochexit = n_epochs
        exit_epoch.append(epochexit)
        # va_accs_alltrials.append(va_acc) 
        # tr_accs_alltrials.append(tr_acc)
        # lowest_alltrials.append(lowest_metric)

        print(f"{p}TRAINING OF {exp_id} COMPLETE{s} (total {spent:.2f} sec, {spent/epochexit:.3f} sec/epoch, {len(train_loader.dataset)*epochexit/spent:.0f} trees/sec)", flush=True)
          
        ###################### CHECK IF FINAL MODEL IS THE BEST (possible that last few training epochs werent done before last val epoch; depends on val_epoch and n_epochs) #########################
   
        va_ys, va_preds = run_utils.test(val_loader, model, n_targ, predict_dist)
        if predict_dist: 
            va_yhats = va_preds[0]
        else:
            va_yhats = va_preds                
        val_metric = np.std(va_yhats - va_ys, axis=0) # sigma for each targ  # metric(va_ys, va_pred) 
        val_rmse = np.sqrt(np.mean((va_ys-va_yhats)**2, axis=0))
        if np.mean(val_metric) < np.mean(best_metric): # Check if current "best" model (avg acc over all targs is better than previous best). NOTE: previously was using criterion that metric was better for only at least one target
            best_metric = val_metric
            best_rmse = val_rmse
            best_preds = va_preds; #best_ys = va_ys
            print(f"New best model is final model; saving in {modelfile}", flush=True)
            torch.save(model.state_dict(), modelfile)
        if not low_output:
            print(f'\tVal metric (scatter): {np.round(val_metric,4)}, lowest was {np.round(best_metric,4)}', flush=True) 

        ###################### LOAD IN BEST MODEL AND "TEST" ON VAL (should give same results as best_metric; if not something is wrong) #########################
        
        # Perform testing 
        model.load_state_dict(torch.load(modelfile))
        ys, preds = run_utils.test(val_loader, model, n_targ, predict_dist) #  WAS THIS AN ISSUE??? run_utils.test(val_loader, model, n_targ, hyper_params['get_sig'], hyper_params['get_cov'])
        if predict_dist: # If predicting mu and sig, test returns [pred_mu, pred_sig]
            yhats = preds[0]
        else:
            yhats = preds
        if eval(str(run_params['performance_plot'])) != None:
            figs = performance_plot(ys, yhats, targ_names)
            for fig, label in zip(figs, targ_names): 
                fig.savefig(f'{out_dir_n}/performance_ne{n_epochs}_{label}.png')
        final_testonval_preds = preds # if predicting mu and sig, this is [pred_mu, pred_sig]
        final_testonval_ys = ys
        final_testonval_metric = np.std(ys-yhats, axis=0)
        final_testonval_rmse = np.sqrt(np.mean((ys-yhats)**2, axis=0))
        # preds_alltrials.append(pred)
        print(f"\n{p}FINAL VALIDATION TEST RMSE{s} FOR {exp_id} : {np.round(final_testonval_rmse, 4)}", flush=True)
        if not np.all(np.round(final_testonval_metric, 4) == np.round(best_metric, 4)): 
            print(f"WARNING: Final test metric ({np.round(final_testonval_metric, 4)}) does not match best metric ({np.round(best_metric, 4)})")
        
        ###################### SAVE PARAMS AND RESULTS (for this trial) #########################
  
        if eval(str(log)):

            # Make params and metrics dicts to add to writer
            paramsf = dict(list(data_params.items()) + list(run_params.items()) + list(hyper_params.items()) + list(learn_params.items()))
            paramsf["targets"] = str(data_params['use_targs']) 
            paramsf["use_feats"] = str(data_params["use_feats"])
            N_p = sum(p.numel() for p in model.parameters())
            N_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
            paramsf['N_params'] = N_p
            paramsf['N_trainable'] = N_t
            from six import string_types
            for k, v in paramsf.items():  # before adding hparams to write, must change any lists to string
                if isinstance(v, list):
                    v_new = '['+''.join(f'{x} ' for x in v).rstrip()+']'
                    paramsf[k] = v_new
                if (isinstance(v, int) or isinstance(v, float) or isinstance(v, string_types) or isinstance(v, bool) or isinstance(v, torch.Tensor) or isinstance(v, list)) == False: 
                    raise ValueError(f'paramsf({k}) is {v} ({type(v)}) -> must be int, float, string, bool, Tensor, or list')  
            #last20 = np.median(va_metrics[-(20//val_epoch):], axis=0);  last10 = np.median(va_metrics[-(10//val_epoch):], axis=0)
            metricf = {'epoch_exit':epoch}
            metricf[f'best_metric'] = best_metric
            # for i in range(n_targ):
            #     metricf[f'scatter_{targ_names[i]}'] = final_testonval_metric[i]
            #     metricf[f'lowest_{targ_names[i]}'] = lowest_metric[i]
            #     metricf[f'last20_{targ_names[i]}'] = last20[i]
            #     metricf[f'last10_{targ_names[i]}'] = last10[i]
            writer.add_hparams(paramsf, metricf, run_name=run_name_n) 
        
        # Collect results into dict
        result_dict={'tot loss each epoch': tot_losses,             # [n_epochs]
                     'sig loss each epoch': sig_losses,             # [n_epochs]
                     'err loss each epoch': err_losses,             # [n_epochs]
                     'lr each val epoch': lrs,                      # [n_epochs/val_epoch]    
                     'val metric each val epoch': va_metrics,       # [n_epochs/val_epoch, n_targs]
                     'train metric each val epoch': tr_metrics,     # [n_epochs/val_epoch, n_targs]
                     'val rmse each val epoch': va_rmses,           # [n_epochs/val_epoch, n_targs]
                     'train rmse each val epoch': tr_rmses,         # [n_epochs/val_epoch, n_targs]
                     'val metric final test': final_testonval_metric,  # [n_targs] # same as RMSE
                     'val rmse final test': final_testonval_rmse,   # [n_targs]
                     'val preds final test': final_testonval_preds, # [[n_val_samples, n_targs], [n_val_samples, n_targs]] 
                     'val ys final test': final_testonval_ys,       # [n_val_samples, n_targs]
                      #'ys': ys, 'pred': pred, 'low_ys': low_ys, 'low_pred': low_pred, 'vars': vars, 'rhos': rhos,  # Why is it usefull to save these at all? 
                     'best val metric': best_metric, # SHOULD BE THE EXACT SAME AS FINAL_TESTONVAL_METRIC
                     'best val rmse': best_rmse, # SHOULD BE THE EXACT SAME AS FINAL_TESTONVAL_RMSE
                     'best val preds': best_preds, # SHOULD BE THE EXACT SAME AS FINAL_TESTONVAL_PREDS
                     'epochexit': epochexit,
                     'training time': t0-time.time(),
                     'gpu': gpu}
        
        # Save results to pickle 
        print(f"{p}SAVING RESULTS FOR {exp_id} TO {resfile}", flush=True)
        pickle.dump(result_dict, open(resfile, 'wb'))
        
    if n_trials > 1: print(f'\nFINISHED ALL {n_trials} TRIALS', flush=True)
    

def train(epoch, schedule, model, train_loader, run_params, gpu, n_targ, loss_func, accelerator, optimizer, scheduler):
    '''
    Train for one epoch
    NOTE: When using uncorrelated guassian loss funcs, model returns 
            sig = predicted standard deviations (aka sig)
         When using correlated guassion loss funcs, model returns
            sig = predicted standard deviations (aka sig)
            cov = predicted covariances (aka rho?)
    '''
    
    model.train()

    l1_lambda = run_params['l1_lambda']
    l2_lambda = run_params['l2_lambda']

    if gpu: init = torch.cuda.FloatTensor([0])
    else: init = torch.FloatTensor([0])
    predict_dist = True if run_params['loss_func'] in ["GaussNd", "Navarro", "GaussNd_corr", "GaussNd_torch"] else False # sig OR cov 
    er_loss = torch.clone(init)
    si_loss = torch.clone(init)
    #rho_loss = torch.tensor([np.nan])

    return_loss = 0
    for data in train_loader: 
        if not gpu:
            data = data.to('cpu')
        if predict_dist:
            muhat, logsighat = model(data)  # var is either sig OR cov
            #if np.any(torch.isnan(logsighat)): raise ValueError('NaNs in predicted SD')
            sighat = torch.exp(logsighat) # assume prediction is of log var instead of var to ensure positivity once exp is taken
            if run_params['loss_func'] == 'GaussNd_torch': 
                loss = loss_func(muhat, data.y.view(-1,n_targ), sighat**2) # torch GaussianNLLLoss expects var, not sig
                err_loss = np.NaN; sig_loss = np.NaN
            else:
                loss, err_loss, sig_loss = loss_func(muhat, data.y.view(-1,n_targ), sighat) 
            er_loss+=err_loss
            si_loss+=sig_loss
        else:
            out = model(data)  
            loss = loss_func(out, data.y.view(-1,n_targ))
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) 
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm # add terms to penalize large param values to loss (?)
        return_loss += loss
        if gpu:
            accelerator.backward(loss)
        else: 
            loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
        if schedule == "onecycle":
            scheduler.step(epoch)

    return return_loss, er_loss, si_loss, l1_lambda * l1_norm, l2_lambda * l2_norm           


def print_results_and_log(log, loss_name, epoch, lr_curr, trainloss, err_loss, sig_loss, l1_loss, l2_loss, train_metric, val_metric, best_metric, counter, n_targ, targ_names, writer=None):
    # Add val results to TensorBoard logger
    if log:
        writer.add_scalar('train_loss', trainloss,global_step=epoch+1)
        writer.add_scalar('learning_rate', lr_curr, global_step=epoch+1)
        writer.add_scalar('best_metric', str(best_metric), global_step=epoch+1)
        # for i in range(n_targ):
        #     #writer.add_scalar(f'last10val_{targ_names[i]}', last10val[i], global_step=epoch+1) 
        #     writer.add_scalar(f'train_scatter_{targ_names[i]}', train_metric[i], global_step=epoch+1)
        #     writer.add_scalar(f'val_scatter_{targ_names[i]}', val_metric[i], global_step=epoch+1) 
        #     writer.add_scalar(f'best_scatter_{targ_names[i]}', best_metrics[i], global_step=epoch+1)
    
    # Print train and val results 
    print(f'\nEpoch {int(epoch+1)} (learning rate {lr_curr:.2E})', flush=True)
    print(f'\tTrain losses: total = {trainloss.cpu().detach().numpy():.2E},  L1 regularization = {l1_loss.cpu().detach().numpy():.2E},  L2 regularization = {l2_loss.cpu().detach().numpy():.2E}', flush=True)
    print(f'\tTrain metric (scatter): {np.round(train_metric,4)}', flush=True)
    if loss_name in ["Gauss1d", "Gauss2d", "Gauss2d_corr", "Gauss4d_corr", "Gauss_Nd", "GaussNd_torch"]:
        print(f'\t[Err/Sig/Rho]: {err_loss.cpu().detach().numpy()[0]:.2E}, {sig_loss.cpu().detach().numpy()[0]:.2E}',  flush=True)       
    print(f'\tVal metric (scatter): {np.round(val_metric,4)}, lowest was {np.round(best_metric,4)}', flush=True) 
    print(f'\t{counter} epochs since last improvement', flush=True) #print(f'\tMedian metric (scatter) for last 10 val epochs: {np.round(last10val,4)} ({counter} epochs since last improvement)', flush=True)
    
    return
    

# def get_metrics(metric_name):
#     '''
#     Returns functions to scrutinize performance on the fly
#     '''

#     import dev.metrics as metrics
#     metrics=getattr(metrics, metric_name)
#     return metrics


def get_performance_plot(name):
    '''
    Load plotting function from plot_utils file
    '''
    import dev.plot_utils as evals
    if name not in [None, "None"]:
        performance_plot = getattr(evals, name)
    else:
        performance_plot = None

    return performance_plot 

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