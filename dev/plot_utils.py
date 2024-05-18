
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib as mpl
import scipy.stats as stats
import pickle, json
import networkx as nx
from itertools import count
import torch
import torch_geometric as tg
from torch_geometric.data import Data
import os, sys
import pandas as pd
try: 
    from dev import data_utils
except:
    sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
    from ObservablesFromTrees.dev import data_utils

def multi_base(ys, pred, targets):
    ''' 
    ys/pred should be the same dimensionality, targets should be numerical indexed, not boolean
    '''
    n_t = len(targets)
    figs=[]
    for n in range(n_t):
        fig, ax =plt.subplots(1,2, figsize=(12,6))
        ax=ax.flatten()
        ax[0].plot(ys[:,n],pred[:,n], 'ro', alpha=0.3)
        ax[0].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[0].set(xlabel='Simulation Truth',ylabel='GNN Prediction', title=targets[n])
        yhat=r'$\hat{y}$'
        ax[0].text(0.6,0.15, f'Bias (mean(y-{yhat})) : {np.mean(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].text(0.6,0.1, 'rmse :  '+f'{np.std(ys[:,n]-pred[:,n]):.3f}', transform=ax[0].transAxes)
        ax[0].legend()
        vals, x, y, _ =ax[1].hist2d(ys[:,n],pred[:,n],bins=50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.magma)
        X, Y = np.meshgrid((x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2)
        ax[1].contour(X,Y, np.log(vals.T+1), levels=10, colors='black')
        ax[1].plot([min(ys[:,n]),max(ys[:,n])],[min(ys[:,n]),max(ys[:,n])], 'k--', label='Perfect correspondance')
        ax[1].set(xlabel='Truth',ylabel='GNN Prediction', title=targets[n])
        ax[1].legend()
        fig.tight_layout()
        figs.append(fig)
    return figs

def plot_preds(ys, preds, labels, name):

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for i in range(2):
        for j in range(3):
            idx = 3*i+j
            pred = preds[:,idx] 
            y = ys[:,idx]   
            #vals, x_pos, y_pos, hist = axs[i,j].hist2d(pred, y, bins=50, range=[np.percentile(np.hstack([pred,y]), [0,100]), np.percentile(np.hstack([pred,y]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
            vals, x_pos, y_pos, hist = axs[i,j].hist2d(y, pred, bins=50, range=[np.percentile(np.hstack([y,pred]), [0,100]), np.percentile(np.hstack([y,pred]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis, alpha=0.5)
            X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
            axs[i,j].contour(X,Y, np.log(vals.T+1), levels=4, linewidths=0.75, colors='white')
            axs[i,j].plot([np.min(y),np.max(y)],[np.min(y),np.max(y)], 'k--', label='Perfect correspondance')
            axs[i,j].text(0.02, 0.95, f'{labels[idx]}', fontsize=10, transform=axs[i,j].transAxes, weight='bold')
            axs[i,j].text(0.6, 0.2, f'rmse: {np.round(float(np.std(pred-y)),3)}', fontsize=8, transform=axs[i,j].transAxes)
            rho = stats.pearsonr(y, pred).statistic # = np.cov(np.stack((y, pred), axis=0))/(np.std(y)*np.std(pred)) # pearson r
            axs[i,j].text(0.6, 0.15, f'pearson r: {np.round(float(rho),3)}', fontsize=8, transform=axs[i,j].transAxes)
        
    fig.supxlabel(f'Hydro Truth')
    fig.supylabel(f'Model Prediction')
    fig.suptitle(name)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    return fig

def plot_preds_compare(ys, preds, sm_ys, sm_preds, targs):

    labels = {' mBH black hole mass [1.0E09 Msun](27)':'M_BH [1.0E09 Msun]',
              ' mcold cold gas mass in disk [1.0E09 Msun](15)': 'M_gas [1.0E09 Msun]',
              ' Metal_cold metal mass in cold gas [Zsun*Msun] (20)': 'M_met [Zsun*Msun]',
              'mstar stellar mass [1.0E09 Msun](8)': 'M_star [1.0E09 Msun]',
              ' sfrave100myr SFR averaged over 100 Myr [Msun/yr](23)': 'SFR_avg [Msun/yr]',
              ' sfr instantaneous SFR [Msun/yr](21)': 'SFR_inst [Msun/yr]'}

    fig, axs = plt.subplots(6, 2, figsize=(8, 15), sharey='row')
    for i in range(len(targs)):
        pred = preds[:,i]; sm_pred = sm_preds[:,i]
        y = ys[:,i]; sm_y = sm_ys[:,i]
        vals, x_pos, y_pos, hist = axs[i,0].hist2d(y, pred, bins=50, range=[np.percentile(np.hstack([y,pred]), [0,100]), np.percentile(np.hstack([y,pred]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis, alpha=0.5)
        X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
        axs[i,0].contour(X,Y, np.log(vals.T+1), levels=4, linewidths=0.75, colors='white')
        axs[i,0].plot([np.min(y),np.max(y)],[np.min(y),np.max(y)], 'k--', label='Perfect correspondance')
        #axs[i,0].text(0.02, 0.95, f'{labels[i]}', fontsize=10, transform=axs[i,0].transAxes, weight='bold')
        axs[i,0].set_ylabel(f'{labels[targs[i]]}', fontsize=9)
        axs[i,0].text(0.6, 0.2, f'rmse: {np.round(float(np.std(pred-y)),3)}', fontsize=8, transform=axs[i,0].transAxes)
        axs[i,0].tick_params(axis='both', which='major', labelsize=8)
        rho = stats.pearsonr(y, pred).statistic # = np.cov(np.stack((y, pred), axis=0))/(np.std(y)*np.std(pred)) # pearson r
        axs[i,0].text(0.6, 0.15, f'pearson r: {np.round(float(rho),3)}', fontsize=8, transform=axs[i,0].transAxes)
        vals, x_pos, y_pos, hist = axs[i,1].hist2d(sm_y, sm_pred, bins=50, range=[np.percentile(np.hstack([sm_y,sm_pred]), [0,100]), np.percentile(np.hstack([sm_y,sm_pred]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis, alpha=0.5)
        X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
        axs[i,1].contour(X,Y, np.log(vals.T+1), levels=4, linewidths=0.75, colors='white')
        axs[i,1].plot([np.min(sm_y),np.max(sm_y)],[np.min(sm_y),np.max(sm_y)], 'k--', label='Perfect correspondance')
        axs[i,1].text(0.6, 0.2, f'rmse: {np.round(float(np.std(sm_pred-sm_y)),3)}', fontsize=8, transform=axs[i,1].transAxes)
        rho = stats.pearsonr(sm_y, sm_pred).statistic # = np.cov(np.stack((y, pred), axis=0))/(np.std(y)*np.std(pred)) # pearson r
        axs[i,1].text(0.6, 0.15, f'pearson r: {np.round(float(rho),3)}', fontsize=8, transform=axs[i,1].transAxes)
        axs[i,1].tick_params(axis='both', which='major', labelsize=8)

    axs[0,0].set_title('GNN', fontsize=10)
    axs[0,1].set_title('MLP', fontsize=10) 
    fig.supxlabel(f'Hydro Truth', fontsize=12)
    fig.supylabel(f'Model Prediction', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.15)

    return fig

def plot_phot_preds(ys, preds, labels, unit, name, sample=False):

    l1, l2, step = -25, -12, 2
    u = '[mag]'
    if unit == 'erg/s/cm²/Å':
        l1, l2 , step = -10, -5, 1
        u = r'log($f_{\lambda}$) [erg/s/cm²/Å]'
    yhats, sampled = data_utils.get_yhats(preds, sample)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    plt.xticks(np.arange(l1, l2, step))
    plt.yticks(np.arange(l1, l2, step))
    for i in range(2):
        for j in range(4):
            idx = 4*i+j
            plot_phot(axs[i,j], yhats, ys, idx, labels, unit, l1, l2)  
    #if sample: axs[0, -1].text(0.5, 0.85, 'NOTE: all data sampled from \npredicted distributions', fontsize=6, transform=axs[0, -1].transAxes)
    note = 'sampled from predicted distributions' if sampled  else 'using predicted means'
    plt.suptitle(f'{name} U-V color vs r magnitude {note}')
    fig.supxlabel(f'Hydro Truth {u}')
    fig.supylabel(f'Model Prediction {u}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_phot_preds_compare(taskdir, models, unit = 'erg/s/cm²/Å', sample=False, uselabels=None):

    l1, l2 = -25, -12
    rmse_min = 0.5; rmse_max = 1.3
    u = '[mag]'
    ticks = np.arange(l1, l2, 3)
    if unit == 'erg/s/cm²/Å':
        l1, l2 = -10, -5
        rmse_min = 0.21; rmse_max = 0.38
        u = r'log($f_{\lambda}$) [erg/s/cm²/Å]'
        ticks = np.arange(l1, l2, 1)
    if sample: l2 += 1
    if uselabels == None: # if not specified, use all labels (and get from any model's config - currently assuming all predict same targets)
        configfile = f'{taskdir}/MLP/{models[0]}_config.json' if 'MLP' in models[0] else f'{taskdir}/{models[0]}/expfile.json'
        uselabels = json.load(open(configfile, 'rb'))['data_params']['use_targs']
    fig, axs = plt.subplots(len(uselabels), len(models), figsize=(4*len(models), len(uselabels)*2.5), sharex=True, sharey=True)
    textcmap = truncate_colormap(cm.get_cmap('hot'), 0, 0.3) #truncate_colormap(cm.get_cmap('ocean'), 0, 0.2)
    plt.xticks(ticks)
    plt.yticks(ticks)
    for j in range(len(models)):
        resfile = f'{taskdir}/MLP/{models[j]}_testres.pkl' if 'MLP' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        modelname = models[j] if 'MLP' in models[j] else f'GNN {models[j]}'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats, sampled = data_utils.get_yhats(preds, sample)
        for i in range(len(uselabels)):
            plot_phot(axs[i,j], yhats, ys, i, uselabels, unit, l1, l2, textcmap, rmse_min, rmse_max)  
        axs[0,j].set_title(f'{modelname}', fontsize=10)
        if sample and len(preds)!=2: axs[j,0].text(0.1, 0.6, 'PREDICTED MEANS \n(NOT A DIST \nMODEL)', fontsize=8, transform=axs[0,j].transAxes)
    fig.supxlabel(f'Hydro Truth {u}')
    fig.supylabel(f'Model Prediction {u}')
    note = 'sampled from predicted distributions' if sampled else 'using predicted means'
    plt.suptitle(f'U-V color vs r magnitude {note}\n')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_phot(ax, yhats, ys, idx, labels, unit, l1, l2, textcmap=None, rmse_min=None, rmse_max=None):

    Fr = [0.79, -0.09, -0.02, -0.40, 8.9e-5, 8.8e-5, 8.7e-5, 8.28e-5] # SDSS U, B,V, in erg/s/cm²/Å. approx u, g, r, i, z conversions from nano-maggies
    yhat = yhats[:,idx] 
    y = ys[:,idx]   
    if unit == 'erg/s/cm²/Å':
        y = (y - Fr[idx])/2.5 # 10**((y - Fr[idx])/2.5)
        yhat = (yhat - Fr[idx])/2.5 # 10**((pred - Fr[idx])/2.5)
    vals, x_pos, y_pos, hist = ax.hist2d(y, yhat, bins=50, range=[np.percentile(np.hstack([y,yhat]), [0,100]), np.percentile(np.hstack([y,yhat]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
    X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
    ax.contour(X,Y, np.log(vals.T+1), levels=4, linewidths=0.75, colors='white')
    ax.plot([l1,l2],[l1,l2], 'k--', label='Perfect correspondance')
    ax.text(0.1, 0.9, f'{labels[idx]} band', fontsize=8, transform=ax.transAxes, weight='bold')
    rmse = np.round(float(np.std(yhat-y)),3)
    rmse_color = textcmap(1-(rmse-rmse_min)/(rmse_max-rmse_min)) if textcmap is not None else 'black'
    ax.text(0.6, 0.15, f'rmse: {rmse}', fontsize=10, transform=ax.transAxes, color=rmse_color)
    rho = np.round(float(stats.pearsonr(y, yhat).statistic ),3) # = np.cov(np.stack((y, yhat, axis=0))/(np.std(y)*np.std(yhat)) # pearson r
    rho_color = textcmap((rho-0.9)/(0.95-0.9)) if textcmap is not None else 'black'
    ax.text(0.6, 0.2, f'pearson r: {rho}', fontsize=10, transform=ax.transAxes, color=rho_color)
    ax.set_xlim(l1, l2)
    ax.set_ylim(l1, l2)

    return ax


def plot_UVr_compare(taskdir, models, sample=False):

    fig, axs = plt.subplots(1, len(models)+1, figsize=((len(models)+1)*3, 4), sharey=True)
    ys, _, _ = pickle.load(open(f'{taskdir}/exp3/testres.pkl', 'rb')) # get test ys from any model (all have same test set)
    Us = ys[:,0]; Vs = ys[:,2]; rs = ys[:,5]
    extent = [[-23, -14], [-4, 4]] # approx limits of worst model (MLP3) when sampled - idk if this really captures things well though
    trueUVr, xedges, yedges = np.histogram2d(rs, Us-Vs, bins=100, range=extent, density=True)  # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
    axs[0].plot(rs, Us-Vs, 'ko', markersize=2, alpha=0.1)
    axs[0].set(ylabel=f' U-V color', xlabel='r magnitude', title='True')
    for j in range(len(models)):
        if 'MLP' in models[j]:
            resfile = f'{taskdir}/MLP/{models[j]}_testres.pkl'
            modelname = models[j]
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
            modelname = f'GNN {models[j][:4]}'
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats, sampled = data_utils.get_yhats(preds, sample)
        Uhats = yhats[:,0]; Vhats = yhats[:,2]; rhats = yhats[:,5]
        predUVr, xedges, yedges = np.histogram2d(rhats, Uhats-Vhats, bins=100, range=extent, density=True) # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
        predUVr = np.where(predUVr==0, 0.001, predUVr) # add small value to zero bins, otherwise KL_div will be inf for those bins # # COULD TRY instead normalizing only after histograming, and normalize to not exactly std normal to add some small values predUVr = (predUVr - np.mean(predUVr))/np.std(predUVr) 
        kldiv = np.round(np.mean(special.kl_div(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same
        relentr = np.round(np.mean(special.rel_entr(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same
        #print(kldiv); fig, axs = plt.subplots(1, 2); im1 = axs[0].imshow(trueUVr); plt.colorbar(im1); im2 = axs[1].imshow(predUVr); plt.colorbar(im2); plt.show(); a=b
        # ks_stat, ks_Pval = np.round(stats.ks_2samp(trueUVr, predUVr), 2) # Tests H0: two dists are the same. High P_value = cannot reject H0
        mmd = np.round(float(run_utils.MMD(trueUVr, predUVr, kernel='rbf')), 3) # Tests H0: two dists are the same. High P_value = cannot reject H0  
        t = axs[j+1].text(0.6, 0.15, f'KLD: {kldiv}\nRE: {relentr}\nMMD: {mmd}', fontsize=10, transform=axs[j+1].transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
        axs[j+1].plot(rhats, Uhats-Vhats, 'ko', markersize=2, alpha=0.1)
        axs[j+1].set_title(f'{modelname}', fontsize=10)
        axs[j+1].set_xlabel(f'r magnitude')
        #if sample and len(preds)==2: axs[j+1].text(0.5, 0.8, 'NOTE: data sampled from \npredicted distributions', fontsize=6, transform=axs[j+1].transAxes)
    note = 'sampled from predicted distributions' if sampled else 'using predicted means'
    plt.suptitle(f'U-V color vs r magnitude {note}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig

def plot_phot_err(ys, preds, labels, unit, name, sample=False):

    Fr = [0.79, -0.09, -0.02, -0.40, 8.9e-5, 8.8e-5, 8.7e-5, 8.28e-5] # SDSS U, B,V, in erg/s/cm²/Å. approx u, g, r, i, z conversions from nano-maggies
    l1, l2, step = -25, -12, 2
    u = '[mag]'
    if unit == 'erg/s/cm²/Å':
        l1, l2, step = -10, -5, 1
        u = r'log($f_{\lambda}$) [erg/s/cm²/Å]'
    yhats, sampled = data_utils.get_yhats(preds, sample)
    fig, axs = plt.subplots(2, 4, figsize=(15, 5), sharex=True, sharey=True)
    plt.xticks(np.arange(l1, l2, step))
    plt.yticks(np.linspace(-30, 30, 5))
    for i in range(2):
        for j in range(4):
            idx = 4*i+j
            yhat = yhats[:,idx] 
            y = ys[:,idx]   
            if unit == 'erg/s/cm²/Å':
                y = (y - Fr[idx])/2.5 # 10**((y - Fr[idx])/2.5)
                yhat = (yhat- Fr[idx])/2.5 # 10**((pred - Fr[idx])/2.5)
            rel_err = 100*(yhat - y)/y
            bin_means, bin_edges, binnumber = stats.binned_statistic(y, rel_err, statistic='mean', bins=40)
            bin_99th, _, _ = stats.binned_statistic(y, rel_err, statistic=lambda rel_err: np.percentile(rel_err, 99), bins=40)
            bin_1st, _, _ = stats.binned_statistic(y, rel_err, statistic=lambda rel_err: np.percentile(rel_err, 1), bins=40)
            bin_75th, _, _ = stats.binned_statistic(y, rel_err, statistic=lambda rel_err: np.percentile(rel_err, 75), bins=40)
            bin_25th, _, _ = stats.binned_statistic(y, rel_err, statistic=lambda rel_err: np.percentile(rel_err, 25), bins=40)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            axs[i,j].plot(bin_centers, bin_means, 'g')
            axs[i,j].fill_between(bin_centers, bin_25th, bin_75th, color='g', alpha=0.2, label='50%')
            axs[i,j].fill_between(bin_centers, bin_1st, bin_99th, color='g', alpha=0.07, label='98%')
            axs[i,j].axhline(0, color='black', linestyle='--')
            axs[i,j].text(0.1, 0.9, f'{labels[idx]} band', fontsize=8, transform=axs[i,j].transAxes, weight='bold')
            axs[i,j].set_xlim(l1, l2)
            axs[i,j].set_ylim(-30, 30)
    #if sampled: axs[0, -1].text(0.6, 0.8, 'NOTE: all data sampled from \npredicted distributions', fontsize=6, transform=axs[0, -1].transAxes)
    note = 'using samples from predicted distributions' if sampled  else 'using predicted means'
    plt.suptitle(f'{name} percent error {note}')
    axs[-1, -1].legend()   
    fig.supxlabel(f'Hydro Truth {u}')
    fig.supylabel('Percent Error')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def plot_err_epoch(val_rmse, targs, stop_epoch, units, name):
    '''
    Should be able to use this for any type of target
    '''
    if not isinstance(units, list):
        units = [units]*len(targs)
        Fr = [0.79, -0.09, -0.02, -0.40, 8.9e-5, 8.8e-5, 8.7e-5, 8.28e-5] # SDSS U, B,V, in erg/s/cm²/Å. approx u, g, r, i, z conversions from nano-maggies
        c = ['indigo','blue','green','red','firebrick','brown','maroon','black']
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    for i in range(len(targs)):
        rmse = val_rmse[:,i]
        if units[i] == 'erg/s/cm²/Å':
            rmse = 10**((rmse - Fr[i])/2.5) # rmse_f = Fr[i]*10**((-rmse)/2.5) # convert to f_lambda (ASSUMING THIS IS ABS MAG)
            ve = np.linspace(0, stop_epoch, len(rmse), dtype=int)
            ax.plot(ve, rmse, color=c[i], label=targs[i], alpha=0.5)
    # ax.set_xticklabels(np.linspace(0, n_epoch, 6, dtype=int))
    ax.set_yscale('log')
    ax.legend()
    ax.set_ylabel(f'Validation Error [{units[i]}]')
    ax.set_xlabel("Epoch")
    fig.suptitle(name)

    return fig 

def plot_colormags(ys, preds, targs, cs, name):

    fig, axs = plt.subplots(len(cs), 2, figsize=(4*2, 3*len(cs)), sharey='row')
    for i in range(len(cs)):  
        c1_idx = targs.index(cs[i][0])
        c2_idx = targs.index(cs[i][1])
        U_true = ys[:,c1_idx]
        V_true = ys[:,c2_idx]
        U_pred = preds[:,c1_idx]
        V_pred = preds[:,c2_idx]
        axs[i,0].plot(V_true, U_true - V_true, 'ko', markersize=2, alpha=0.1)
        axs[i,1].plot(V_pred, U_pred - V_pred, 'ko', markersize=2, alpha=0.1)
        axs[i,0].set(ylabel=f' {cs[i][0]} - {cs[i][1]} color', xlabel=f'{cs[i][1]} magnitude')
        axs[i,1].set_xlabel(f'{cs[i][1]} magnitude')
    axs[0,0].set_title('True')
    axs[0,1].set_title('Predicted')
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    
    return fig

def plot_UVr(ys, preds, name):

    fig, axs = plt.subplots(1, 2, figsize=(4*2, 4), sharey='row')
    axs[0].plot(ys[:,5], ys[:,0]- ys[:,2], 'ko', markersize=2, alpha=0.1)
    axs[1].plot(preds[:,5], preds[:,0]- preds[:,2], 'ko', markersize=2, alpha=0.1)
    axs[0].set(ylabel=f' U-V color', xlabel='r magnitude')
    axs[1].set_xlabel(f'r magnitude')
    axs[0].set_title('True')
    axs[1].set_title('Predicted')
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0)
    
    return fig

def plot_color_merghist_compare(taskdir, models, sample=False):

    ys, _, _ = pickle.load(open(f'{taskdir}/exp3/testres.pkl', 'rb')) # get test ys from any model (all have same test set)
    U_true = ys[:,0]; V_true = ys[:,2]
    Vbins = [-22, -21, -20, -17, -13]
    fig, axs = plt.subplots(len(Vbins)-1, len(models)+1, figsize=((len(models)+1)*3, 1.5*len(Vbins)), sharey=True, sharex=True)

    for i in range(len(Vbins)-1):
        Utrue = U_true[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        Vtrue = V_true[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        MHmetrics = pickle.load(open('../Data/vol100/DS1_[0, 1, 3, 4]_[0, 1, 2, 3, 4, 5, 6, 7]_[70, 10, 20]_testMHmetrics.pkl', 'rb'))
        z_50 = np.array(MHmetrics['z_50'].values)
        z_50_true = z_50[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        axs[i,0] = get_binned_dist(axs[i,0], z_50_true,  Utrue - Vtrue,  nstatbins = np.unique(z_50_true))
        axs[i,0].set_ylabel('U-V color')
        axs[-1,0].set_xlabel('z50')
    axs[0,0].set_title('Truth')

    for j in range(len(models)):
        if 'MLP' in models[j]:
            # resfile = f'{taskdir}/MLP/{models[j]}_testres.pkl'
            # modelname = models[j]
            raise NameError('Cant yet add MLPs because cant go back and create MH metrics since xs are not full graphs. And cant use the same as for the GNN models becasue used random TTS. If need to do this, will need to at least re-test the MLPs on final halos data created from the same graphs in the GNN test set.')
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
            MHmetrics = pickle.load(open('../Data/vol100/DS1_[0, 1, 3, 4]_[0, 1, 2, 3, 4, 5, 6, 7]_[70, 10, 20]_testMHmetrics.pkl', 'rb'))
            modelname = f'GNN {models[j][:4]}'
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats, sampled = data_utils.get_yhats(preds, sample) 
        Uhats = yhats[:,0]; Vhats = yhats[:,2]  
        for i in range(len(Vbins)-1):
            Uhat = Uhats[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            Vhat = Vhats[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            z_50 = np.array(MHmetrics['z_50'].values)
            z_50_pred = z_50[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            axs[i,j+1] = get_binned_dist(axs[i,j+1], z_50_pred,  Uhat - Vhat,  nstatbins = np.unique(z_50_pred))
            axs[0,j+1].set_title(modelname)
            axs[-1,j+1].set_xlabel('z50')
            axs[i,int((len(models)+1)/2)].text(0.75, 0.85, f'V mag bin {Vbins[i]} to {Vbins[i+1]} ', transform=axs[i,1].transAxes, backgroundcolor='w')
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig 


def plot_color_merghist(ys, preds, MHmetrics, name, sample=False):

    metric_names = ['Total N Halos', r'$z_{25}$', r'$z_{50}$', r'$z_{75}$']
    z_50 = np.array(MHmetrics['z_50'].values)
    U_true = ys[:,0]; V_true = ys[:,2]
    yhats, sampled = data_utils.get_yhats(preds, sample)
    Uhats = yhats[:,0]; Vhats = yhats[:,2]
    bins = [-22, -21, -20, -17, -13]
    fig, axs = plt.subplots(len(bins)-1, 2, figsize=(3*2, 1.5*len(bins)), sharey=True, sharex=True)
    for i in range(len(bins)-1):

        Utrue = U_true[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        Vtrue = V_true[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        z_50_true = z_50[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        axs[i,0] = get_binned_dist(axs[i,0], z_50_true,  Utrue - Vtrue,  nstatbins = np.unique(z_50_true))
        axs[i,0].set_ylabel('U-V color')

        Uhat = Uhats[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        Vhat = Vhats[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        z_50_pred = z_50[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        axs[i,1] = get_binned_dist(axs[i,1], z_50_pred,  Uhat - Vhat, nstatbins = np.unique(z_50_pred))
        axs[i,1].text(-0.25, 0.85, f'V mag bin {bins[i]} to {bins[i+1]} ', transform=axs[i,1].transAxes, backgroundcolor='w')

    axs[-1,0].set_xlabel(f'{metric_names[2]}')
    axs[-1,1].set_xlabel(f'{metric_names[2]}')
    axs[0,0].set_title('True')
    axs[0,1].set_title('Predicted')
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig      

def get_binned_dist(ax, x, y, nstatbins): 

    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=nstatbins)
    bin_99th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 99), bins=nstatbins)
    bin_1st, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 1), bins=nstatbins)
    bin_75th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 75), bins=nstatbins)
    bin_25th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 25), bins=nstatbins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    ax.plot(bin_centers, bin_means, 'g')
    ax.fill_between(bin_centers, bin_25th, bin_75th, color='g', alpha=0.2, label='50%')
    ax.fill_between(bin_centers, bin_1st, bin_99th, color='g', alpha=0.07, label='98%')

    return ax

def plot_phot_errby_halofeat(ys, preds, z0xall, used_feats, all_feats, all_featdescriptions, tree_meta, plot_by, name):

    z0xall = np.vstack(z0xall)
    rel_err = 100*(ys - preds)/ys
    avg_rel_err = np.mean(rel_err, axis = 1) # for each obs, avg rel err across all bands

    logcols = tree_meta['logcols']
    minmaxcols = tree_meta['minmaxcols']

    nrows = int(np.ceil(len(plot_by)/4))
    fig, axs = plt.subplots(nrows, 4, figsize=(5*4, 3*nrows), sharey=True)
    for i in range(nrows):
        for j in range(4):
            feat_idx = all_feats.index(plot_by[i*4+j])
            feat = z0xall[:,feat_idx]
            nbins = 25
            bin_means, bin_edges, binnumber = stats.binned_statistic(feat, avg_rel_err, statistic='mean', bins=nbins)
            bin_99th, _, _ = stats.binned_statistic(feat,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 99), bins=nbins)
            bin_1st, _, _ = stats.binned_statistic(feat,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 1), bins=nbins)
            bin_75th, _, _ = stats.binned_statistic(feat,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 75), bins=nbins)
            bin_25th, _, _ = stats.binned_statistic(feat, avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 25), bins=nbins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            axs[i,j].plot(bin_centers, bin_means, 'g')
            axs[i,j].fill_between(bin_centers, bin_25th, bin_75th, color='g', alpha=0.2, label='50%')
            axs[i,j].fill_between(bin_centers, bin_1st, bin_99th, color='g', alpha=0.07, label='98%')
            axs[i,j].axhline(0, color='black', linestyle='--')
            if plot_by[i*4+j] == 'Rvir(11)': axs[i,j].set_xlim(40, 275)
            if plot_by[i*4+j] == 'Spin_Bullock': axs[i,j].set_xlim(0, 0.25)
            if plot_by[i*4+j] == 'T/|U|': axs[i,j].set_xlim(0.5, 0.93)
            if plot_by[i*4+j] == 'vrms(13)': axs[i,j].set_xlim(25, 225)
            if plot_by[i*4+j] in logcols:
                feat_label = f'log {all_featdescriptions[plot_by[i*4+j]]}'
            else:
                feat_label = all_featdescriptions[plot_by[i*4+j]]
            axs[i,j].set(xlabel=f'{feat_label}')
        axs[i,0].set(ylabel=r'Avg. $\%$ Error (all bands)')
    axs[-1,-1].legend()
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)

    return fig

def plot_phot_errby_galtarg(ys, preds,  all_testtargs, all_targdescriptions, plot_by, name):

    rel_err = 100*(ys - preds)/ys
    avg_rel_err = np.mean(rel_err, axis = 1) # for each obs, avg rel err across all bands
    nrows = int(np.ceil(len(plot_by)/4))
    fig, axs = plt.subplots(nrows, 4, figsize=(5*4, 3*nrows), sharey=True)
    for i in range(nrows):
        for j in range(4):
            targ = np.array(all_testtargs[plot_by[i*4+j]], dtype=float)
            nbins = 25
            bin_means, bin_edges, binnumber = stats.binned_statistic(targ, avg_rel_err, statistic='mean', bins=nbins)
            bin_99th, _, _ = stats.binned_statistic(targ,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 99), bins=nbins)
            bin_1st, _, _ = stats.binned_statistic(targ,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 1), bins=nbins)
            bin_75th, _, _ = stats.binned_statistic(targ,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 75), bins=nbins)
            bin_25th, _, _ = stats.binned_statistic(targ, avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 25), bins=nbins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            axs[i,j].plot(bin_centers, bin_means, 'b')
            axs[i,j].fill_between(bin_centers, bin_25th, bin_75th, color='b', alpha=0.2, label='50%')
            axs[i,j].fill_between(bin_centers, bin_1st, bin_99th, color='b', alpha=0.07, label='98%')
            axs[i,j].axhline(0, color='black', linestyle='--')
            feat_label = all_targdescriptions[plot_by[i*4+j]]
            axs[i,j].set(xlabel=f'{feat_label}')
            if plot_by[i*4+j] == 'SubhaloSFR': axs[i,j].set_xlim(0, 8)
            if plot_by[i*4+j] == 'SubhaloStelMass': axs[i,j].set_xlim(0,5)
        axs[i,0].set(ylabel=r'Avg. $\%$ Error (all bands)')
    axs[-1,-1].legend()
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)

    return fig

def plot_phot_errby_merghist(ys, preds, test_MHmetrics, name):

    rel_err = 100*(ys - preds)/ys
    avg_rel_err = np.mean(rel_err, axis = 1) # for each obs, avg rel err across all bands
    fig, axs = plt.subplots(1, 4, figsize=(5*4, 3), sharey=True)
    metric_names = ['Total N Halos', r'$z_{25}$', r'$z_{50}$', r'$z_{75}$']
    for i in range(4):
        metric = test_MHmetrics[list(test_MHmetrics.keys())[i]] 
        nbins = 12
        bin_means, bin_edges, binnumber = stats.binned_statistic(metric, avg_rel_err, statistic='mean', bins=nbins)
        bin_99th, _, _ = stats.binned_statistic(metric,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 99), bins=nbins)
        bin_1st, _, _ = stats.binned_statistic(metric,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 1), bins=nbins)
        bin_75th, _, _ = stats.binned_statistic(metric,  avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 75), bins=nbins)
        bin_25th, _, _ = stats.binned_statistic(metric, avg_rel_err, statistic=lambda  avg_rel_err: np.percentile(avg_rel_err, 25), bins=nbins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        axs[i].plot(bin_centers, bin_means, 'r')
        axs[i].fill_between(bin_centers, bin_25th, bin_75th, color='r', alpha=0.2, label='50%')
        axs[i].fill_between(bin_centers, bin_1st, bin_99th, color='r', alpha=0.07, label='98%')
        axs[i].axhline(0, color='black', linestyle='--')
        feat_label = metric_names[i]
        axs[i].set(xlabel=f'{feat_label}')
    axs[0].set(ylabel=r'Avg. $\%$ Error (all bands)')
    axs[-1].legend()
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig    

def plot_spatial(pos_test, mstar_test, pos_raw, mstar_raw):

    fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
    im = axs[0].scatter(pos_test[:,0]/0.704, pos_test[:,1]/0.704, c=mstar_test, s=2, alpha=0.5)
    axs[0].set_title('Test set galaxies')
    axs[0].set_ylabel('x pos Mpc')
    axs[0].set_xlabel('y pos Mpc')
    mstar_raw1 = np.where(np.isinf(mstar_raw), np.NaN, mstar_raw) # set -inf (from taking log of zero) to np.NaN
    im = axs[1].scatter(pos_raw[:,0]/0.704, pos_raw[:,1]/0.704, c=mstar_raw1, s=0.5, alpha=0.5)
    axs[1].set_title('TNG galaxies')
    axs[1].set_xlabel('x pos Mpc')
    fig.subplots_adjust(right=0.7)
    cbar_ax = fig.add_axes([0.9999, 0.15, 0.01, 0.7])
    fig.tight_layout()
    fig.colorbar(im, cax=cbar_ax, label=r'log$_{10}$ M$_{*}$')
    plt.subplots_adjust(wspace=0, hspace=0.3)

    return fig

def plot_PS_band_compare(taskdir, dataset, models = ['exp3', 'exp4', 'exp6'], by='V mag', sample=True, plotdiff=False):

    nc = 128
    bs = 110.7*0.704 # want in units of Mpc/h
    if by == 'V mag':
        bins = [-22, -21, -20, -18, -15, -13]
    if by == 'U-V':
        raise NotImplementedError('Set bins U-V!')
    colors = cm.get_cmap(truncate_colormap(cm.get_cmap('jet'), 0.3, 0.7))(np.linspace(0,1,len(bins)-1)) # pl.cm.jet(np.linspace(0,1,len(Vbins)-1))
    #print(cm.get_cmap(colors)(np.linspace(0,1,len(Vbins)-1)))

    # Load test data information (pos, Mstar)
    testdata, test_z0xall = pickle.load(open(f'{taskdir}/exp3/testdata.pkl', 'rb')) # get test ys from any model (all have same test set)
    featnames = json.load(open(f"../Data/vol100/{dataset}_meta.json", 'rb'))[2]['tree meta']['featnames']
    z0xall = np.vstack(test_z0xall)
    pos = np.vstack([z0xall[:, featnames.index('x(17)')], z0xall[:, featnames.index('y(18)')], z0xall[:, featnames.index('z(19)')]]).T # Mpc/h
    all_otherSHtargs = pickle.load(open(f'../Data/vol100/subfind_othertargs.pkl', 'rb'))
    testidxs = np.load(osp.expanduser(f"../Data/vol100/{dataset}_testidx20.npy")) 
    all_testtargs =  pd.DataFrame(columns = all_otherSHtargs.columns)
    for i in range(len(all_otherSHtargs)):
        if i in testidxs: all_testtargs.loc[len(all_testtargs)] = all_otherSHtargs.iloc[i]
    mstar = np.log10(np.array(all_testtargs['SubhaloStelMass'], dtype=float)*(10**10)/0.704) # mstar for test set (e.g. transformed)
    ys, _, _ = pickle.load(open(f'{taskdir}/exp3/testres.pkl', 'rb')) 
    V_true = ys[:,2]; U_true = ys[:,0]
    
    # If not plotting difference, plot for true bins
    if plotdiff:
        fig, axs = plt.subplots(1, len(models), figsize=((len(models))*4, 4), sharey=True, sharex=True)
    else:
        fig, axs = plt.subplots(1, len(models)+1, figsize=((len(models)+1)*4, 4), sharey=True, sharex=True)
        for i in range(len(bins)-1):
            if by == 'V mag':
                idxs = np.where(V_true < bins[i+1])[0] # np.where((val > bins[i]) & (val < bins[i+1]))[0]
            elif by == 'U-V':
                idxs = np.where((V_true-U_true  > bins[i]) & (V_true-U_true  < bins[i+1]))[0]
            pos_true_bin = pos[idxs]
            pos_true_bin_nc = (pos_true_bin/bs) * nc # want in units of nc
            mstar_true_bin = mstar[idxs]  # Dont normalize - should allow bins with more massive gals to plot higher up # weights = mstar_true_bin/len(mstar_true_bin) # lets normalize by num gals in that bin, but not by the max mass in the bin #np.max(mstar_true_bin)
            field_true = paintcic(pos_true_bin_nc, mass=mstar_true_bin, bs=bs, nc=nc)
            field_true = field_true/np.mean(field_true)
            k_true, p_true = power(field_true, boxsize=bs) 
            p_true_scl = p_true #- sum(mstar_true_bin)/bs**3 # sum of all masses in bin divided by volume
            axs[0].plot(k_true, p_true_scl, c=colors[i])
        axs[0].set_xlabel(r'k [$h^{-1} Mpc$]')
        axs[0].set_title('Truth')

    # Plot for pred bins
    for j in range(len(models)):
        if 'MLP' in models[j]:
            resfile = f'{taskdir}/MLP/{models[j]}_testres.pkl'
            modelname = models[j]
            model_ds = json.load(open(f"{taskdir}/MLP/{modelname}_config.json","rb"))['data_params']['data_file'].replace('.pkl', '')
            #raise NameError('Cant yet add MLPs because use different test set, so pos and mstar dont correspond. If need to do this, will need to at least re-test the MLPs on final halos data created from the same graphs in the GNN test set.')
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
            modelname = f'GNN {models[j][:4]}'
            model_ds = json.load(open(f"{taskdir}/{models[j]}/expfile.json","rb"))['data_params']['data_file'].replace('.pkl', '')
        if dataset not in model_ds:
                raise ValueError(f'{modelname} was trained on {model_ds} not {dataset}. Shouldnt compare models run on different datasets')
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats, sampled = data_utils.get_yhats(preds, sample) 
        V_pred = yhats[:,2]; U_pred = yhats[:,0] 
        ax_idx = j if plotdiff else j+1
        for i in range(len(bins)-1):
            if by == 'V mag':
                idxs = np.where(V_pred < bins[i+1])[0] # np.where((val > bins[i]) & (val < bins[i+1]))[0]
            elif by == 'U-V':
                idxs = np.where((V_pred-U_pred  > bins[i]) & (V_pred-U_pred  < bins[i+1]))[0]
            pos_bin = pos[idxs]
            pos_pred_bin_nc = (pos_bin/bs) * nc # want in units of nc
            mstar_bin = mstar[idxs]  
            field_pred = paintcic(pos_pred_bin_nc, mass=mstar_bin, bs=bs, nc=nc)
            field_pred = field_pred/np.mean(field_pred)
            k_pred, p_pred = power(field_pred, boxsize=bs)
            p_pred_scl = p_pred - sum(mstar_bin)/bs**3 # sum of all masses in bin divided by volume
            if plotdiff:
                if by == 'V mag':
                    idxs = np.where(V_true < bins[i+1])[0] # np.where((val > bins[i]) & (val < bins[i+1]))[0]
                elif by == 'U-V':
                    idxs = np.where((V_true-U_true  > bins[i]) & (V_true-U_true  < bins[i+1]))[0]
                pos_true_bin = pos[idxs]
                pos_true_bin_nc = (pos_true_bin/bs) * nc # want in units of nc
                mstar_true_bin = mstar[idxs]  # Dont normalize - should allow bins with more massive gals to plot higher up # weights = mstar_true_bin/len(mstar_true_bin) # lets normalize by num gals in that bin, but not by the max mass in the bin #np.max(mstar_true_bin)
                field_true = paintcic(pos_true_bin_nc, mass=mstar_true_bin, bs=bs, nc=nc)
                field_true = field_true/np.mean(field_true)
                k_true, p_true = power(field_true, boxsize=bs) 
                p_true_scl = p_true #- sum(mstar_true_bin)/bs**3 # sum of all masses in bin divided by volume
                label = f'{by} {bins[i]} to {bins[i+1]}' if by == 'U-V' else f'{by} < {bins[i+1]}'
                axs[ax_idx].plot(k_pred, p_pred_scl-p_true_scl, label=label, c=colors[i])   
            else:
                axs[ax_idx].plot(k_pred, p_pred_scl, label=f'{by} {bins[i]} to {bins[i+1]}', c=colors[i])   
            axs[ax_idx].set_xlabel(r'k [$h^{-1} Mpc$]')
            axs[ax_idx].set_title(modelname)

    axs[-1].legend()
    xlab = r'predicted - true p(k) [$(h^{-1} Mpc)^3$]' if plotdiff else r'p(k) [$(h^{-1} Mpc)^3$]' 
    axs[0].set_ylabel(xlab)
    plt.semilogx(); #plt.semilogy()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(top=0.85)
    note = 'sampled from predicted distributions' if sampled  else 'using predicted means'
    plt.suptitle(f'Power spectra {note}')

    return fig


###########
# Graph plotting functions
###########


def plot_tree(graph, scalecol, masscol, fig, ax, first=False, last=False, colorbymass=True, node_ids='None'):

    ####
    # Is k mass? That would seem like a good chance to color nodes by, and its what he has is colorbar titled
    #   k=3 for christian's set, and if he made his set like mine (all cols except id cols), or with the defaul tcols, col 3 is Mvir
    # But then why was he setting the di node attribute to have the name 'n_prog'? Just a typo?
    # And for his paper figure he uses node size and color to represent mass
    ####
    
    x = graph.x.numpy()
    G = tg.utils.to_networkx(graph)

    # posy=[]
    # a_vals, counts = np.unique(x[:,scalecol], return_counts=True)
    # for a in x[:,scalecol]:
    #     posy.append(np.where(a==a_vals)[0][0]) 
    posy = 106.667*x[:,scalecol] - 6.667

    hpos = hierarchy_pos(G.reverse())
    for p in hpos.keys():
        hpos[p] = [hpos[p][0]*1.05, -posy[p]]
    nodes = G.nodes()
    
    if colorbymass:
        di = nx.betweenness_centrality(G)
        featr = []
        for q, key in enumerate(di.keys()):
            feat = x[q][masscol] #feat = transformer[tkeys[k]].inverse_transform(graph.x.numpy()[q,k].reshape(-1, 1))[0][0]
            featr.append(feat)
            di[key] = feat
        nx.set_node_attributes(G, di, 'Mvir') # just a typo that he was calling this n_prog??
        labels = nx.get_node_attributes(G, 'Mvir') # dict of  "index: value" pairs 
        masses = set(labels.values()) # just a typo that he was calling this progs??
        node_size = (x[:,masscol]-min(x[:,3])+1) # **1.2
        mapping = dict(zip(sorted(masses),count())) # should be dict of   # sorted() sorts by first value (scale)
        #colors = [G.nodes[n]['Mvir'] for n in nodes] 
        colors = [mapping[G.nodes[n]['Mvir']] for n in nodes]
    elif not isinstance(node_ids, str): 
        colors = ['w' for n in nodes]
    else:
        colors = ['b' for n in nodes]


    # label_zpos = np.unique(1/x[:,scalecol] - 1) #  scale = 1/(1+z)    # np.unique(1/transformer[0].inverse_transform(feats[:,0].reshape(-1,1))-1)
    label_ypos = np.linspace(0, -100, 20)
    labels = np.round(0.11*label_ypos + 11) # 0.11*label_ypos - 11, 1)
    ec = nx.draw_networkx_edges(G, hpos, alpha=0.9, width=0.5, arrows=False, ax=ax)

    if not isinstance(node_ids, str):
        node_ids = pd.Series([id[:-3] for id in node_ids])
        nc = nx.draw_networkx_nodes(G, hpos, nodelist=nodes, node_color=colors, node_size=30, ax=ax)
        nx.draw_networkx_labels(G, hpos, labels=node_ids, font_size=8, ax=ax)
    elif colorbymass:
        nc = nx.draw_networkx_nodes(G, hpos, nodelist=nodes, node_color=colors, node_size=(node_size), cmap=plt.cm.jet, ax=ax)
    else:
        nc = nx.draw_networkx_nodes(G, hpos, nodelist=nodes, node_color=colors, node_size=30, ax=ax)

    if first:
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)
        ax.set(yticks = label_ypos) #ax.set(yticks=-np.linspace(0, max(zs), 100)[1::6]) #ax.set(yticks=-np.arange(len(zs))[1::6])
        ax.set_yticklabels(labels) #ax.set_yticklabels(np.round(zs[:-1],1)[::-6])
        ax.set(ylabel='Redshift')
    if last:
        if not first: ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=True)
        if isinstance(node_ids, str) and colorbymass:
            rs = np.round(np.percentile(featr, np.linspace(0,100,8)),1)
            cbar = fig.colorbar(nc, shrink=0.5, ax=ax)
            cbar.ax.set_yticklabels(rs)
            cbar.set_label(r'Halo mass [log($M_h/M_{\odot}$)]') # cbar.set_label('Redshift')
    ax.set_xticks([])

    return fig


def plot_trees_compare(tree_list, labels_list, featnames=['#scale(0)','desc_scale(2)','num_prog(4)','Mvir(10)']):

    fig, axs = plt.subplots(1, len(tree_list), figsize=(len(tree_list)*7, 7), layout='constrained', sharey=True)


    for i in range(len(tree_list)): 
        axs[i].set_title(labels_list[i])
        tree = tree_list[i]
        if len(tree) > 12000: 
            axs[i].text(0.2,0.5, f"Too many halos for efficient plotting ({len(tree)})", transform=axs[i].transAxes)
            continue
        data = np.array(tree[featnames], dtype=float)
        X = torch.tensor(data, dtype=torch.float) 
        y = torch.tensor(np.nan, dtype=torch.float) 
        edge_index, edge_attr = data_utils.make_edges(tree)
        graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
        fig = plot_tree(graph, scalecol=0, masscol=3, fig=fig, ax=axs[i], first=True)

    return fig

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])

    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos

def plot_tree_OLD(graph, scalecol, masscol, fig, ax, first=False, last=False):
    ######
    # OLD
    ######
    
    x = graph.x.numpy()
    G = tg.utils.to_networkx(graph)
    # posy=[]
    # a_vals, counts = np.unique(x[:,scalecol], return_counts=True)
    # for a in x[:,scalecol]:
    #     posy.append(np.where(a==a_vals)[0][0]) 
    posy = 106.667*x[:,scalecol] - 6.667
    hpos = hierarchy_pos_OLD(G.reverse())
    for p in hpos.keys():
        hpos[p] = [hpos[p][0]*1.05, -posy[p]]
    di = nx.betweenness_centrality(G)
    featr = []
    for q, key in enumerate(di.keys()):
        feat = x[q][masscol] #feat = transformer[tkeys[k]].inverse_transform(graph.x.numpy()[q,k].reshape(-1, 1))[0][0]
        featr.append(feat)
        di[key] = feat
    nx.set_node_attributes(G, di, 'Mvir') # just a typo that he was calling this n_prog??
    labels = nx.get_node_attributes(G, 'Mvir') # dict of  "index: value" pairs 
    masses = set(labels.values()) # just a typo that he was calling this progs??
    mapping = dict(zip(sorted(masses),count())) # should be dict of   # sorted() sorts by first value (scale)
    nodes = G.nodes()
    #colors = [G.nodes[n]['Mvir'] for n in nodes] 
    colors = [mapping[G.nodes[n]['Mvir']] for n in nodes]
    # label_zpos = np.unique(1/x[:,scalecol] - 1) #  scale = 1/(1+z)    # np.unique(1/transformer[0].inverse_transform(feats[:,0].reshape(-1,1))-1)
    label_ypos = np.linspace(0, -100, 20)
    labels = np.round(0.11*label_ypos + 11) # 0.11*label_ypos - 11, 1)
    rs = np.round(np.percentile(featr, np.linspace(0,100,8)),1)
    #ec = nx.draw_networkx_edges(G, hpos, alpha=0.9, width=0.5, arrows=False, ax=ax)
    node_size = (x[:,masscol]-min(x[:,3])+1) # **1.2
    nc = nx.draw_networkx_nodes(G, hpos, nodelist=nodes, node_color=colors, node_size=(node_size), cmap=plt.cm.jet, ax=ax)

    if first:
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=True)
        ax.set(yticks = label_ypos) #ax.set(yticks=-np.linspace(0, max(zs), 100)[1::6]) #ax.set(yticks=-np.arange(len(zs))[1::6])
        ax.set_yticklabels(labels) #ax.set_yticklabels(np.round(zs[:-1],1)[::-6])
        ax.set(ylabel='Redshift')
    if last:
        if not first: ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=True)
        cbar = fig.colorbar(nc, shrink=0.5, ax=ax)
        cbar.ax.set_yticklabels(rs)
        cbar.set_label(r'Halo mass [log($M_h/M_{\odot}$)]') # cbar.set_label('Redshift')
    ax.set_xticks([])
    return fig

def hierarchy_pos_OLD(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):
    '''
    OLD

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  
    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)
    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.
    **width** horizontal space allocated for this branch - avoids overlap with other branches
    **vert_gap** gap between levels of hierarchy
    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))
    def _hierarchy_pos_OLD(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments
        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos_OLD(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves
            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count
    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos_OLD(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos       

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap      

############################################
# Clustering analysis functions from Chirag
# THIS SHOULDNT REALLY GO HERE
############################################

def power(f1, f2=None, boxsize=1.0, k = None, symmetric=True, demean=True, eps=1e-9):
    """
    Calculate power spectrum given density field in real space & boxsize.
    Divide by mean, so mean should be non-zero
    """
    if demean and abs(f1.mean()) < 1e-3:
        #print('Add 1 to get nonzero mean of %0.3e'%f1.mean())
        f1 = f1*1 + 1
    if demean and f2 is not None:
        if abs(f2.mean()) < 1e-3:
            #print('Add 1 to get nonzero mean of %0.3e'%f2.mean())
            f2 =f2*1 + 1
    
    if symmetric: c1 = np.fft.rfftn(f1)
    else: c1 = np.fft.fftn(f1)
    if demean : c1 /= c1[0, 0, 0].real
    c1[0, 0, 0] = 0
    if f2 is not None:
        if symmetric: c2 = np.fft.rfftn(f2)
        else: c2 = np.fft.fftn(f2)
        if demean : c2 /= c2[0, 0, 0].real
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    if k is None:
        k = fftk(f1.shape, boxsize, symmetric=symmetric)
        k = sum(kk**2 for kk in k)**0.5
    H, edges = np.histogram(k.flat, weights=x.flat, bins=f1.shape[0]) 
    N, edges = np.histogram(k.flat, bins=edges)
    center= edges[1:] + edges[:-1]
    power = H *boxsize**3 / N
    power[power == 0] = np.NaN
    return 0.5 * center,  power

def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return kvector given a shape (nc, nc, nc) and boxsize 
    """
    k = []
    for d in range(len(shape)):
        kd = np.fft.fftfreq(shape[d])
        kd *= 2 * np.pi / boxsize * shape[d]
        kdshape = np.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k

def paintcic(pos, bs, nc, mass=1.0, period=True):
    mesh = np.zeros((nc, nc, nc))
    transform = lambda x: x/bs*nc
    if period: period = int(nc)
    else: period = None
    return paint(pos, mesh, weights=mass, transform=transform, period=period)

def paint(pos, mesh, weights=1.0, mode="raise", period=None, transform=None):
    """ CIC approximation (trilinear), painting points to Nmesh,
        each point has a weight given by weights.
        This does not give density.
        pos is supposed to be row vectors. aka for 3d input
        pos.shape is (?, 3).

        pos[:, i] should have been normalized in the range of [ 0,  mesh.shape[i] )

        thus z is the fast moving index

        mode can be :
            "raise" : raise exceptions if a particle is painted
             outside the mesh
            "ignore": ignore particle contribution outside of the mesh
        period can be a scalar or of length len(mesh.shape). if period is given
        the particles are wrapped by the period.

        transform is a function that transforms pos to mesh units:
        transform(pos[:, 3]) -> meshpos[:, 3]
    """
    pos = np.array(pos)
    chunksize = 1024 * 16 * 4
    Ndim = pos.shape[-1]
    Np = pos.shape[0]
    if transform is None:
        transform = lambda x:x
    neighbours = ((np.arange(2 ** Ndim)[:, None] >> \
            np.arange(Ndim)[None, :]) & 1)
    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        if np.isscalar(weights):
          wchunk = weights
        else:
          wchunk = weights[chunk]
        gridpos = transform(pos[chunk])
        rmi_mode = 'raise'
        intpos = np.intp(np.floor(gridpos))

        for i, neighbour in enumerate(neighbours):
            neighbour = neighbour[None, :]
            targetpos = intpos + neighbour

            kernel = (1.0 - np.abs(gridpos - targetpos)).prod(axis=-1)
            add = wchunk * kernel
            #print(f"   i: {i}, wchunk {wchunk}")#add[:5] = {add[:5]}")

            if period is not None:
                period = np.int32(period)
                np.remainder(targetpos, period, targetpos)

            if len(targetpos) > 0:
                targetindex = np.ravel_multi_index(
                        targetpos.T, mesh.shape, mode=rmi_mode)
                u, label = np.unique(targetindex, return_inverse=True)
                mesh.flat[u] += np.bincount(label, add, minlength=len(u))
                if np.isnan(sum(mesh.flat[u])):
                    raise ValueError(f"NaN being added to mesh")
            
            if np.isnan(np.sum(mesh)):
                raise ValueError(f"NaNs in mesh")

    return mesh

