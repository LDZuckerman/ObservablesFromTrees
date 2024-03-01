#########################################
## Current GPU train loop              ##
## Own libraries loaded in the bottom  ##
#########################################

import torch, pickle, time, os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from importlib import __import__
import random
import string
from tqdm import tqdm

mus, scales = np.array([-1.1917865,  1.7023178,  -0.14979358, -2.5043619]), np.array([0.9338901, 0.17233825, 0.5423821, 0.9948792])

t_labels = np.array(['halo_index (long) (0)',
 'birthhaloid (long long)(1)',
 'roothaloid (long long)(2)',
 'redshift(3)',
 'sat_type 0= central(4)',
 'mhalo total halo mass [1.0E09 Msun](5)',
 'm_strip stripped mass [1.0E09 Msun](6)',
 'rhalo halo virial radius [Mpc)](7)',
 'mstar stellar mass [1.0E09 Msun](8)',
 'mbulge stellar mass of bulge [1.0E09 Msun] (9)',
 ' mstar_merge stars entering via mergers] [1.0E09 Msun](10)',
 ' v_disk rotation velocity of disk [km/s] (11)',
 ' sigma_bulge velocity dispersion of bulge [km/s](12)',
 ' r_disk exponential scale radius of stars+gas disk [kpc] (13)',
 ' r_bulge 3D effective radius of bulge [kpc](14)',
 ' mcold cold gas mass in disk [1.0E09 Msun](15)',
 ' mHI cold gas mass [1.0E09 Msun](16)',
 ' mH2 cold gas mass [1.0E09 Msun](17)',
 ' mHII cold gas mass [1.0E09 Msun](18)',
 ' Metal_star metal mass in stars [Zsun*Msun](19)',
 ' Metal_cold metal mass in cold gas [Zsun*Msun] (20)',
 ' sfr instantaneous SFR [Msun/yr](21)',
 ' sfrave20myr SFR averaged over 20 Myr [Msun/yr](22)',
 ' sfrave100myr SFR averaged over 100 Myr [Msun/yr](23)',
 ' sfrave1gyr SFR averaged over 1 Gyr [Msun/yr](24)',
 ' mass_outflow_rate [Msun/yr](25)',
 ' metal_outflow_rate [Msun/yr](26)',
 ' mBH black hole mass [1.0E09 Msun](27)',
 ' maccdot accretion rate onto BH [Msun/yr](28)',
 ' maccdot_radio accretion rate in radio mode [Msun/yr](29)',
 ' tmerge time since last merger [Gyr] (30)',
 ' tmajmerge time since last major merger [Gyr](31)',
 ' mu_merge mass ratio of last merger [](32)',
 ' t_sat time since galaxy became a satellite in this halo [Gyr](33)',
 ' r_fric distance from halo center [Mpc](34)',
 ' x_position x coordinate [cMpc](35)',
 ' y_position y coordinate [cMpc](36)',
 ' z_position z coordinate [cMpc](37)',
 ' vx x component of velocity [km/s](38)',
 ' vy y component of velocity [km/s](39)',
 ' vz z component of velocity [km/s](40)'])

t_labels = [t.replace(' ','') for t in t_labels]
t_labels = [t.replace('/','') for t in t_labels]
t_labels = np.array(t_labels)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def run(construct_dict, gpu):
    """
    For each of n_trials, train model and then test on test set
    """
    
    # Seperate run params, data params, and learn params
    run_params=construct_dict['run_params']
    data_params=construct_dict['data_params']
    learn_params=construct_dict['learn_params']
    
    # Set output and logging
    out_dir=construct_dict['out_dir']
    out_pointer=osp.expanduser(out_dir)
    if not osp.exists(out_pointer):
        try:
            os.makedirs(out_pointer)
        except:
            print('Folder already exists')
    log=construct_dict['log']
    
    # Set run params
    n_trials=run_params['n_trials']
    n_epochs=run_params['n_epochs']
    val_epoch=run_params['val_epoch']
    batch_size=run_params['batch_size']
    save=run_params['save']
    early_stopping=run_params['early_stopping']
    patience=run_params['patience']
    num_workers=run_params['num_workers']
    if run_params['seed']==True:
        torch.manual_seed(42)
    
    # Set learn params
    lr=learn_params['learning_rate']
    if learn_params['warmup'] and learn_params["schedule"] in ["warmup_exp", "warmup_expcos"]:
        lr=lr/(learn_params['g_up'])**(learn_params['warmup']) # [?] What is this doing?
    lr_scheduler = get_lr_schedule(construct_dict) 
    loss_func = get_loss_func(construct_dict['run_params']['loss_func'])
    l1_lambda=run_params['l1_lambda']
    l2_lambda=run_params['l2_lambda']

    # load data
    train_data, val_data = load_data(**data_params) # WAS CALLING VAL_DATA TEST_DATA BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
    try: 
        n_targ=len(train_data[0].y)
    except:
        n_targ=1
    n_feat=len(train_data[0].x[0])
    val_loader=DataLoader(val_data, batch_size=batch_size, shuffle=0, num_workers=num_workers) # WAS CALLING THIS TEST_LOADER    ##never shuffle test
    construct_dict['hyper_params']['in_channels'] = n_feat
    construct_dict['hyper_params']['out_channels'] = n_targ

    # Set up evaluation metrics 
    metric = get_metrics(construct_dict['run_params']['metrics']) # Returns evaluation function to use
    performance_plot = get_performance(construct_dict['run_params']['performance_plot'])
    train_accs, val_accs, scatter, = [], [], [] # WAS CALLING VAL_ACCS TEST_ACCS BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
    preds,lowest, epochexit = [], [], []
    
    # Perform experiment (either once or n_trials times)
    run_name = f'{construct_dict["id"]}' 
    for trial in range(n_trials):
        
        # Set logging and state-saving dirs
        if n_trials>1:
            run_name_n=run_name+f'_{trial+1}_{n_trials}'
            log_dir=osp.join(out_pointer, run_name_n)
            print(f'Trial {trial}: model outputs and trained model will be saved in {log_dir}/')
        else:
            run_name_n=run_name
            log_dir=osp.join(out_pointer, run_name_n) # e.g. /ObsFromTrees_2024/Task_samProps/exp1/
            print(f'Model outputs and trained model will be saved in {log_dir}/')
        if log:
            from torch.utils.tensorboard import SummaryWriter # [?] What is TensorBoard?
            writer=SummaryWriter(log_dir=log_dir)
        if save:  
            model_path = log_dir # osp.join(log_dir, "trained_model") 
            # if not osp.isdir(model_path):
            #     os.makedirs(model_path)
        
        # Initialize model, dataloaders, scheduler, and optimizer
        model = setup_model(construct_dict['model'], construct_dict['hyper_params']) # print(f"N_params: {sum(p.numel() for p in model.parameters())}")
        train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) #shuffle training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler=lr_scheduler(optimizer, **learn_params, total_steps=n_epochs*len(train_loader))
       
        # If gpu, feed to accelerator
        if gpu == '1':
            from accelerate import Accelerator 
            accelerator = Accelerator() 
            device = accelerator.device
            _, _, val_loader = accelerator.prepare(model, optimizer, val_loader) # WAS CALLING VAL_LOADER TEST_LOADER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else: accelerator = None
        
        # Set up evaluation metrics
        lowest_metric=np.array([np.inf]*n_targ) # 
        low_ys = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        low_pred = torch.tensor([np.inf]) # added this so that can have get_val_results as a function
        tr_acc, va_acc=[],[]
        early_stop=0  # [?] What exactly is this doing??
        start=time.time()
        k=0
        
        ###################### TRAIN (for this trial) #########################
        
        # Perform training
        print(f'\nBEGINING TRAINING ({n_epochs} epochs total)')
        for epoch in tqdm(range(n_epochs)):
            
            # Train for one epoch
            trainloss, err_loss, sig_loss, rho_loss, l1_loss, l2_loss = train(epoch, learn_params["schedule"], model, train_loader, run_params, gpu, n_targ, l1_lambda, l2_lambda, loss_func, accelerator, optimizer, scheduler)
            
            # Step learning rate scheduler
            if learn_params["schedule"]!="onecycle": # onecycle steps per batch, not epoch
                scheduler.step(epoch)
            
            # If this is an epoch ([?] he means a val epoch??), perform validation
            if (epoch+1)%val_epoch==0: 
                
                # Perform validation and store results # WAS CALLING VAL_METRIC TEST_METRIC BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                train_metric, val_metric, lowest_metric, low_ys, low_pred, k, early_stop = get_val_results(model, run_params, data_params, train_loader, val_loader, optimizer, epoch, metric, k, lowest_metric, save, val_epoch, early_stop, n_targ, low_ys, low_pred, model_path)
                tr_acc.append(train_metric)
                va_acc.append(val_metric)  # WAS CALLING VA_ACC TE_ACC BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                lr0=optimizer.state_dict()['param_groups'][0]['lr']
                last10val=np.median(va_acc[-(10//val_epoch):], axis=0)  # WAS CALLING LAST10VAL LAST10TEST BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                
                # Add results to logger
                if log:
                    writer.add_scalar('train_loss', trainloss,global_step=epoch+1)
                    writer.add_scalar('learning_rate', lr0, global_step=epoch+1)
                    if n_targ==1:
                        writer.add_scalar('last10val', last10val, global_step=epoch+1)  # WAS CALLING LAST10VAL LAST10TEST BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                        writer.add_scalar('train_scatter', train_metric,global_step=epoch+1)
                        writer.add_scalar('val_scatter', val_metric, global_step=epoch+1) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                        writer.add_scalar('best_scatter', lowest_metric, global_step=epoch+1)
                    else:
                        labels = t_labels[data_params["targets"]]
                        for i in range(n_targ):
                            writer.add_scalar(f'last10val_{labels[i]}', last10val[i], global_step=epoch+1) # WAS CALLING LAST10VAL LAST10TEST BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                            writer.add_scalar(f'train_scatter_{labels[i]}', train_metric[i], global_step=epoch+1)
                            writer.add_scalar(f'val_scatter_{labels[i]}', val_metric[i], global_step=epoch+1)
                            writer.add_scalar(f'best_scatter_{labels[i]}', lowest_metric[i], global_step=epoch+1)
                
                # Print results 
                if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "Gauss2d_corr", "Gauss4d_corr", "Gauss_Nd"]:
                    print(f'\nEpoch {int(epoch+1)} (learning rate {lr0:.2E})')
                    print(f'\tTrain loss {trainloss.cpu().detach().numpy():.2E}') 
                    print(f'\t[Err/Sig/Rho]: {err_loss.cpu().detach().numpy()[0]:.2E}, {sig_loss.cpu().detach().numpy()[0]:.2E}, {rho_loss.cpu().detach().numpy()[0]:.2E}')
                    print(f'\tL1 regularization loss: {l1_loss.cpu().detach().numpy():.2E}, L2 regularization loss: {l2_loss.cpu().detach().numpy():.2E}')
                    print(f'\tTrain scatter: {np.round(train_metric,4)}')
                else:
                    print(f'\nEpoch: {int(epoch+1)} (learning rate {lr0:.2E})') 
                    print(f'\tTrain loss: {trainloss.cpu().detach().numpy():.2E}')
                    print(f'\tTrain scatter: {np.round(train_metric,4)}')
                    print(f'\tL1 regularization loss: {l1_loss.cpu().detach().numpy():.2E}, L2 regularization loss: {l2_loss.cpu().detach().numpy():.2E}')
                print(f'\tVal scatter: {np.round(val_metric,4)}, Lowest was {np.round(lowest_metric,4)}')  # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                print(f'Median scatter for last 10 val epochs: {np.round(last10val,4)} ({early_stop} epochs since last improvement)')
        
                # Every 5th val epoch (what is this? "testing" in addition to "validating"?????)
                if (epoch+1)%(int(val_epoch*5))==0 and log:
                    if n_targ==1:
                        ys, pred, xs, Mh, vars, rhos = test(val_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"]) # WAS CALLING VAL_LOADER TEST_LOADER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                        fig=performance_plot(ys,pred, xs, Mh, data_params["targets"])
                        writer.add_figure(tag=run_name_n, figure=fig, global_step=epoch+1)
                    else:
                        labels = t_labels[data_params["targets"]]
                        ys, pred, xs, Mh, vars, rhos = test(val_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"]) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                        figs = performance_plot(ys,pred, xs, Mh, data_params["targets"])
                        for fig, label in zip(figs, labels):
                            writer.add_figure(tag=f'{run_name_n}_{label}', figure=fig, global_step=epoch+1)
                        if early_stopping:
                            if early_stop>patience:
                                print(f'Exited after {epoch+1} epochs due to early stopping')
                                epochexit.append(epoch)
                                break
                    if early_stopping:
                        if early_stop<patience:
                            epochexit.append(epoch)
                        else:
                            epochexit.append(n_epochs)
        stop=time.time()
        spent=stop-start
        val_accs.append(va_acc) # WAS CALLING VAL_ACCS TEST_ACCS BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
        train_accs.append(tr_acc)
        
        if early_stopping and n_epochs>epochexit[-1]:
            pr_epoch=epochexit[-1]
        else:
            pr_epoch=n_epochs
            
        print(f"\nTRAINING COMPLETE (total {spent:.2f} sec, {spent/n_epochs:.3f} sec/epoch, {len(train_loader.dataset)*pr_epoch/spent:.0f} trees/sec)")
        
        ###################### TEST (for this trial) (WTF BUT STILL VAL DATA???) #########################
        
        # Perform testing 
        ys, pred, xs, Mh, vars, rhos = test(val_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"]) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
        if save:
            if n_targ==1:
                label = t_labels[data_params["targets"]]
                figs=performance_plot(ys,pred, xs, Mh, data_params["targets"])
                for fig in figs:
                    fig.savefig(f'{log_dir}/performance_ne{n_epochs}_{label}.png')
            else:
                labels = t_labels[data_params["targets"]]
                ys, pred, xs, Mh, vars, rhos = test(val_loader, model, data_params["targets"], run_params['loss_func'], data_params["scale"]) # WAS CALLING VAL_SCATTER TEST_SCATTER BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                figs = performance_plot(ys,pred, xs, Mh, data_params["targets"])
                for fig, label in zip(figs, labels):
                    fig.savefig(f'{log_dir}/performance_ne{n_epochs}_{label}.png')
        sig=np.std(ys-pred, axis=0)
        scatter.append(sig)
        preds.append(pred)
        lowest.append(lowest_metric)
        last20=np.median(va_acc[-(20//val_epoch):], axis=0) # WAS CALLING VA_ACC TE_ACC BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
        last10=np.median(va_acc[-(10//val_epoch):], axis=0)
        
        print(f'\nTESTING (but on validation set???')
        print(f'\tScatter {np.round(val_metric,4)}')
        
        ###################### SAVE PARAMS AND RESULTS (for this trial) #########################
  
        # Make saveable params
        paramsf=dict(list(data_params.items()) + list(run_params.items()) + list(construct_dict['hyper_params'].items()) + list(construct_dict['learn_params'].items()))
        paramsf["targets"]=int("".join([str(i+1) for i in data_params["targets"]]))
        paramsf["del_feats"]=str(data_params["del_feats"])
        ##adding number of model parameters
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
                raise valueError(f'paramsf({k}) is {v} ({type(v)}) -> must be int, float, string, bool, Tensor, or list')  

        # Make saveable metrics 
        labels = t_labels[data_params["targets"]]
        metricf={'epoch_exit':epoch}
        for i in range(n_targ):
            metricf[f'scatter_{labels[i]}']=sig[i]
            metricf[f'lowest_{labels[i]}']=lowest_metric[i]
            metricf[f'last20_{labels[i]}']=last20[i]
            metricf[f'last10_{labels[i]}']=last10[i]
        
        # Add to writer and save results
        writer.add_hparams(paramsf, metricf, run_name=run_name_n) 
        result_dict={'sigma':scatter,
                     'val_acc': va_acc, # WAS CALLING VA_ACC TE_ACC BUT ITS ONLY TEST IF TEST=1 IN JSON!!!
                     'train_acc': tr_acc,
                     'ys': ys,
                     'pred': pred, 
                     'low_ys': low_ys,
                     'low_pred': low_pred,
                     'vars': vars,
                     'rhos': rhos, 
                     'low':lowest,
                     'epochexit': epochexit}
        with open(f'{log_dir}/result_dict.pkl', 'wb') as handle:
            pickle.dump(result_dict, handle)
        with open(f'{log_dir}/construct_dict.pkl', 'wb') as handle:
            pickle.dump(construct_dict, handle)
    
        print(f'\nFINISHED {trial+1}/{n_trials} TRIALS')
        
###################################################
# Split data into train and val or train and test #
###################################################

def load_data(data_path, data_file, targets, del_feats, scale, test=0, splits=[70, 10, 20], ): ##data_params
    
    print(f'Loading data from {data_path}/{data_file}')
    
    datat=pickle.load(open(osp.expanduser(f'{data_path}/{data_file}'), 'rb'))
    a=np.arange(43)
    feats=np.delete(a, del_feats)
    data=[]
    for d in datat:
        if not scale:
            data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y[targets]))
        else:
            data.append(Data(x=d.x[:, feats], edge_index=d.edge_index, edge_attr=d.edge_attr, y=(d.y[targets]-torch.Tensor(mus[targets]))/torch.Tensor(scales[targets])))
    
    testidx_file = osp.expanduser(f'{data_path}/{data_file.replace(".pkl", "")}_testidx{splits[2]}.npy')
    if not os.path.exists(testidx_file): 
        print(f'Test indices for {splits[2]}% test set not yet saved for this data. Creating and saving in {testidx_file}')
        indices = np.random.choice(np.linspace(0, len(data)-1, len(data), dtype=int), int(len(data)*(splits[2]/100)), replace=False)
        np.save(testidx_file, indices)
    print(f'Loading test indices from {testidx_file}')
    testidx = np.array(np.load(testidx_file))

    test_data=[]
    trainval_data=[] 
    for i, d in tqdm(enumerate(data)):
        if i in testidx:
            test_data.append(d)
        else:
            trainval_data.append(d)
    if test: # if "test" run, which Christian uses to mean training the same model again completely from scratch on train+val, then testing on test
        return trainval_data, test_data # merged train+val, test
    else:
        val_pct = splits[1]/(splits[0]+splits[1]) # pct of train
        train_data = trainval_data[:int(len(trainval_data)*(1-val_pct))]
        val_data = trainval_data[int(len(trainval_data)*(val_pct)):]
        return train_data, val_data  
    
##################################
#       Train one epoch          #
##################################

def train(epoch, schedule, model, train_loader, run_params, gpu, n_targ, l1_lambda, l2_lambda, loss_func, accelerator, optimizer, scheduler):
    model.train()
    return_loss=0
    if gpu =='1':
        er_loss = torch.cuda.FloatTensor([0])
        si_loss = torch.cuda.FloatTensor([0])
        rh_loss = torch.cuda.FloatTensor([0])
    else:
        er_loss = torch.FloatTensor([0])
        si_loss = torch.FloatTensor([0])
        rh_loss = torch.FloatTensor([0]) 
    for data in train_loader: 
        if gpu != '1':
            data = data.to('cpu')
        if run_params["loss_func"] in ["L1", "L2", "SmoothL1"]: 
            out = model(data)  
            loss = loss_func(out, data.y.view(-1,n_targ))
        if run_params["loss_func"] in ["Gauss1d", "Gauss2d", "GaussNd"]:
            out, var = model(data)  
            loss, err_loss, sig_loss = loss_func(out, data.y.view(-1,n_targ), var)
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
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        return_loss+=loss
        if gpu == '1':
            accelerator.backward(loss)
        else: 
            loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
        if schedule=="onecycle":
            scheduler.step(epoch)
    # if epoch==0:  writer.add_graph(model,[data])   #Doesn't work right now but could be fun to add back in

    return return_loss, er_loss, si_loss, rh_loss, l1_lambda * l1_norm, l2_lambda * l2_norm           

################################
#    Validate for one batch    #
################################

def get_val_results(model, run_params, data_params, train_loader, test_loader, optimizer, epoch, metric, k, lowest_metric, save, val_epoch, early_stop, n_targ, low_ys, low_pred, model_path):
   
    if run_params['metrics']!='test_multi_varrho':
        train_metric, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
        test_metric, ys, pred = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
    else:
        train_metric, _, _, _, _ = metric(train_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
        test_metric, ys, pred, vars, rhos = metric(test_loader, model, data_params['targets'], run_params['loss_func'], data_params['scale'])
    if k==0: # If this is first val epoch
        lowest_metric=test_metric
        low_ys = ys
        low_pred = pred
        k+=1
    if np.any(test_metric<lowest_metric): # If this is the current best model
        mask=test_metric<lowest_metric
        index=np.arange(n_targ)[mask]
        lowest_metric[mask]=test_metric[mask]
        low_ys[:,index]=ys[:,index]
        low_pred[:,index]=pred[:,index]
        early_stop=0
        if save:
            torch.save(model.state_dict(), osp.join(model_path,'model_best.pt'))
    else:
        early_stop+=val_epoch

    return train_metric, test_metric, lowest_metric, low_ys, low_pred, k, early_stop
  
############################
#       Test model         #
############################

def test(loader, model, targs, l_func, scale):
    '''returns targets and predictions, and some test metrics
    This function here isn't pretty, xs shouldn't be used, they take up too much memory'''
    ys, pred,xs, Mh=[],[],[],[]
    model.eval()
    n_targ=len(targs)
    outs = []
    ys = []
    vars= []
    rhos = []
    if scale:
        sca=torch.cuda.FloatTensor(scales[targs])
        ms=torch.cuda.FloatTensor(mus[targs])
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
            if scale:
                ys.append(data.y.view(-1,n_targ)*sca+ms)
                pred.append(out*sca+ms)
            else:
                ys.append(data.y.view(-1,n_targ))
                pred.append(out)
            vars.append(var)
            rhos.append(rho)

    ys = torch.vstack(ys)
    pred = torch.vstack(pred)
    vars = torch.vstack(vars)
    rhos = torch.vstack(rhos)
    xn=[] ## keep for downstream dependency
    return ys.cpu().numpy(), pred.cpu().numpy(), xn, Mh, vars, rhos     
    
################################
#      Load dependencies       #
################################

def get_metrics(metric_name):
    '''Returns functions to scrutinize performance on the fly'''
    import dev.metrics as metrics
    metrics=getattr(metrics, metric_name)
    return metrics

################################
#         Custom loss       #
################################

def get_loss_func(name):
    # Return loss func from the loss functions file given a function name
    import dev.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    try:
        l=loss_func()
    except:
        l=loss_func
    return l

def get_performance(name):
    '''Returns plotting functions to scrutinize performance on the fly'''
    import dev.eval_plot as evals
    performance_plot = getattr(evals, name)
    return performance_plot 

def setup_model(model_name, hyper_params):
    # Retrieve name and params for construction

    # Load model from model folder
    import dev.models as models
    model = getattr(models, model_name) # get function [model_name]() from dev/models.py
    model = model(**hyper_params)

    return model

def get_lr_schedule(construct_dict):
    schedule  = construct_dict['learn_params']['schedule']

    import dev.lr_schedule as lr_module

    schedule_class = getattr(lr_module, schedule)

    return schedule_class