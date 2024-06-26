
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib as mpl
import scipy.stats as stats
import pickle, json
import networkx as nx
from itertools import count
from sympy import E
import torch
from torch import Value
import torch_geometric as tg
from torch_geometric.data import Data
import sys
import os.path as osp
import pandas as pd
import scipy.special as special
import arviz as az
try: 
    from dev import data_utils
except:
    sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
    from ObservablesFromTrees.dev import data_utils, run_utils


####################################
# Functions for single model results
####################################

def plot_phot_preds(ys, preds, labels, unit='mag', name=None, sample_method='mean'):
    '''
    Plot predicted vs true values for each phot target
    - If sampling, plot samples from predicted distributions
    '''

    yhats = data_utils.get_yhats(preds, sample_method)
    sighats = preds[1]
    if yhats.ndim == 3:
        yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    # plt.xticks(np.arange(l1, l2, step))
    # plt.yticks(np.arange(l1, l2, step))
    for i in range(2):
        for j in range(4):
            idx = 4*i+j
            plot_targ(axs[i,j], yhats[:,idx], sighats[:,idx], ys[:,idx])  # sighats not used if not sample_method is not no_sample
            axs[i,j].text(0.1, 0.9, f'{labels[idx]} band', fontsize=8, transform=axs[i,j].transAxes, weight='bold')
    #if sample: axs[0, -1].text(0.5, 0.85, 'NOTE: all data sampled from \npredicted distributions', fontsize=6, transform=axs[0, -1].transAxes)
    fig.supxlabel(f'Hydro Truth [mag]')
    fig.supylabel(f'Model Prediction [mag]')
    plt.suptitle(name)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def plot_props_preds(ys, preds, labels, name):
    '''
    Plot predicted vs true values for each prop target
    '''

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for i in range(2):
        for j in range(3):
            idx = 3*i+j
            pred = preds[:,idx] 
            y = ys[:,idx]   
            #vals, x_pos, y_pos, im = axs[i,j].hist2d(pred, y, bins=50, range=[np.percentile(np.hstack([pred,y]), [0,100]), np.percentile(np.hstack([pred,y]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
            vals, x_pos, y_pos, im = axs[i,j].hist2d(y, pred, bins=50, range=[np.percentile(np.hstack([y,pred]), [0,100]), np.percentile(np.hstack([y,pred]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis, alpha=0.5)
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

def plot_phot_err(ys, preds, labels, unit='mag', name=None, sample_method=None):
    '''
    Plot prediction error vs true values for each phot target
    '''

    # Fr = [0.79, -0.09, -0.02, -0.40, 8.9e-5, 8.8e-5, 8.7e-5, 8.28e-5] # SDSS U, B,V, in erg/s/cm²/Å. approx u, g, r, i, z conversions from nano-maggies
    # l1, l2, step = -25, -12, 2
    # u = '[mag]'
    # if unit == 'erg/s/cm²/Å':
    #     l1, l2, step = -10, -5, 1
    #     u = r'log($f_{\lambda}$) [erg/s/cm²/Å]'
    yhats = data_utils.get_yhats(preds, sample_method)
    if yhats.ndim == 3:
        yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
    fig, axs = plt.subplots(2, 4, figsize=(15, 5), sharex=True, sharey=True)
    # plt.xticks(np.arange(l1, l2, step))
    # plt.yticks(np.linspace(-30, 30, 5))
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
            # axs[i,j].set_xlim(l1, l2)
            # axs[i,j].set_ylim(-30, 30)
    #if sampled: axs[0, -1].text(0.6, 0.8, 'NOTE: all data sampled from \npredicted distributions', fontsize=6, transform=axs[0, -1].transAxes)
    note = 'using samples from predicted distributions' if sample_method != None  else 'using predicted means'
    plt.suptitle(f'{name} percent error {note}')
    axs[-1, -1].legend()   
    fig.supxlabel(f'Hydro Truth [mag]')
    fig.supylabel('Percent Error')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def plot_err_epoch(val_rmse, targs, stop_epoch, units, name):
    '''
    Plot validation error over training epoch
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
    '''
    NOT MAINTAINED
    Plot predicted and true color vs magnitude for given color pairs
    '''

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
    '''
    Plot predicted and true U-V color vs r magnitude
    '''

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

def plot_color_merghist(ys, preds, MHmetrics, name, sample_method=None):
    '''
    Plot predicted and true U-V color vs z_50
    '''

    metric_names = ['Total N Halos', r'$z_{25}$', r'$z_{50}$', r'$z_{75}$']
    z_50 = np.array(MHmetrics['z_50'].values)
    U_true = ys[:,0]; V_true = ys[:,2]
    yhats = data_utils.get_yhats(preds, sample_method)
    Uhats = yhats[:,0]; Vhats = yhats[:,2]
    bins = [-22, -21, -20, -17, -13]
    fig, axs = plt.subplots(len(bins)-1, 2, figsize=(3*2, 1.5*len(bins)), sharey=True, sharex=True)
    for i in range(len(bins)-1):

        Utrue = U_true[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        Vtrue = V_true[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        z_50_true = z_50[np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0]]
        axs[i,0] = plot_binned_dist(axs[i,0], z_50_true,  Utrue - Vtrue,  nstatbins = np.unique(z_50_true))
        axs[i,0].set_ylabel('U-V color')

        Uhat = Uhats[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        Vhat = Vhats[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        z_50_pred = z_50[np.where((Vhats > bins[i]) & (Vhats < bins[i+1]))[0]]
        axs[i,1] = plot_binned_dist(axs[i,1], z_50_pred,  Uhat - Vhat, nstatbins = np.unique(z_50_pred))
        axs[i,1].text(-0.25, 0.85, f'V mag bin {bins[i]} to {bins[i+1]} ', transform=axs[i,1].transAxes, backgroundcolor='w')

    axs[-1,0].set_xlabel(f'{metric_names[2]}')
    axs[-1,1].set_xlabel(f'{metric_names[2]}')
    axs[0,0].set_title('True')
    axs[0,1].set_title('Predicted')
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig      

def plot_phot_errby_halofeat(ys, preds, z0xall, used_feats, all_feats, all_featdescriptions, tree_meta, plot_by, name):
    '''
    Plot phot prediction error as a function of halo features
    '''

    z0xall = np.vstack(z0xall)
    rel_err = 100*(ys - preds)/ys
    avg_rel_err = np.mean(rel_err, axis = 1) # for each obs, avg rel err across all bands

    logcols = tree_meta['lognames']
    minmaxcols = tree_meta['minmaxnames']

    nrows = int(np.ceil(len(plot_by)/4))
    ncols = 4 if len(plot_by) > 3 else len(plot_by)
    fig, axs = plt.subplots(nrows, 4, figsize=(5*4, 3*nrows), sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            feat_idx = all_feats.index(plot_by[i*4+j])
            feat = z0xall[:,feat_idx]
            bins = 25
            ax = axs[i,j] if nrows > 1 else axs[j] if ncols > 1 else axs
            ax = plot_binned_dist(ax, feat,  avg_rel_err,  nstatbins = bins)
            ax.axhline(0, color='black', linestyle='--')
            if plot_by[i*4+j] == 'Rvir(11)': ax.set_xlim(40, 275)
            if plot_by[i*4+j] == 'Spin_Bullock': ax.set_xlim(0, 0.25)
            if plot_by[i*4+j] == 'T/|U|': ax.set_xlim(0.5, 0.93)
            if plot_by[i*4+j] == 'vrms(13)': ax.set_xlim(25, 225)
            if plot_by[i*4+j] in logcols:
                feat_label = f'log {all_featdescriptions[plot_by[i*4+j]]}'
            else:
                feat_label = all_featdescriptions[plot_by[i*4+j]]
            ax.set(xlabel=f'{feat_label}')
        ax = axs[i,0] if nrows > 1 else axs[0] if ncols > 1 else axs
        ax.set(ylabel=r'Avg. $\%$ Error (all bands)')
    axs.flatten()[-1].legend() if ncols > 1 else axs.legend()
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)

    return fig

def plot_phot_errby_galtarg(ys, preds,  all_testtargs, all_targdescriptions, plot_by, name):
    '''
    Plot phot prediction error as a function of galaxy properties
    '''

    rel_err = 100*(ys - preds)/ys
    avg_rel_err = np.mean(rel_err, axis = 1) # for each obs, avg rel err across all bands
    nrows = int(np.ceil(len(plot_by)/4))
    ncols = 4 if len(plot_by) > 3 else len(plot_by)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            targ_label = all_targdescriptions[plot_by[i*4+j]]
            targ = np.array(all_testtargs[plot_by[i*4+j]], dtype=float)
            bins = 25
            if plot_by[i*4+j] == 'SubhaloStelMass':
                targ = np.log10(np.array(targ, dtype=float)*(10**10)/0.704)
                targ_label = r'log $M_{\star}$ [$M_{\odot}$]'
            ax = axs[i,j] if nrows > 1 else axs[j] if ncols > 1 else axs
            ax = plot_binned_dist(ax, targ,  avg_rel_err,  nstatbins = bins)
            ax.axhline(0, color='black', linestyle='--')
            ax.set(xlabel=f'{targ_label}')
            #if plot_by[i*4+j] == 'SubhaloSFR': ax.set_xlim(0, 8)
        ax = axs[i,0] if nrows > 1 else axs[0] if ncols > 1 else axs
        ax.set(ylabel=r'Avg. $\%$ Error (all bands)')
    axs.flatten()[-1].legend() if ncols > 1 else axs.legend()
    fig.suptitle(name)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)

    return fig

def plot_phot_errby_merghist(ys, preds, test_MHmetrics, name):
    '''
    Plot phot prediction error as a function of z_25, z_50, z_75
    '''

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
    '''
    Visualize clustering by plotting simple spatial image of galaxies 
    '''

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

def plot_samprops_preds(ys, preds, labels, targs, sample_method='mean'):
    '''
    Plot predicted vs true values for each target
    '''

    fig, axs = plt.subplots(len(targs), 1, figsize=(4, len(targs)*3), sharey='row')
    for i in range(len(targs)):
        yhats  = data_utils.get_yhats(preds, sample_method)
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        y = ys[:,i]
        yhat = yhats[:,i]
        vals, x_pos, y_pos, im = axs[i].hist2d(y, yhat, bins=50, range=[np.percentile(np.hstack([y,yhat]), [0,100]), np.percentile(np.hstack([y,yhat]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis, alpha=0.5)
        X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
        axs[i].contour(X,Y, np.log(vals.T+1), levels=4, linewidths=0.75, colors='white')
        axs[i].plot([np.min(y),np.max(y)],[np.min(y),np.max(y)], 'k--', label='Perfect correspondance')
        #axs[i,0].text(0.02, 0.95, f'{labels[i]}', fontsize=10, transform=axs[i,0].transAxes, weight='bold')
        axs[i].set_ylabel(f'{labels[targs[i]]}', fontsize=9)
        axs[i].text(0.6, 0.2, f'rmse: {np.round(float(np.std(yhat-y)),3)}', fontsize=8, transform=axs[i].transAxes)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        rho = stats.pearsonr(y, yhat).statistic # = np.cov(np.stack((y, pred), axis=0))/(np.std(y)*np.std(pred)) # pearson r
        axs[i].text(0.6, 0.15, f'pearson r: {np.round(float(rho),3)}', fontsize=8, transform=axs[i].transAxes)

    fig.supxlabel(f'Hydro Truth', fontsize=12)
    fig.supylabel(f'Model Prediction', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.15)

    return fig

#####################################
# Functions for compareing model results
#####################################

def compare_phot_preds(taskdir, models, modelnames, sample_method='no_sample', plottype='heat', N=100, uselabels=None):
    '''
    For given models, compare predicted vs true values for each phot target
    '''

    rmse_min = 0.5; rmse_max = 1.3
    lims = [(-28, -11),(-28, -11)]
    fig, axs = plt.subplots(len(uselabels), len(models), figsize=(4*len(models), len(uselabels)*2.5), sharex=True, sharey=True)
    textcmap = None # truncate_colormap(cm.get_cmap('hot'), 0, 0.3) #truncate_colormap(cm.get_cmap('ocean'), 0, 0.2)
    for j in range(len(models)):
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        sighats = preds[1]
        yhats = data_utils.get_yhats(preds, sample_method, N)
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        for i in range(len(uselabels)):
            plot_targ(axs[i,j], yhats[:,i], sighats[:,i], ys[:,i], lims=lims, textcmap=textcmap, rmse_min=rmse_min, rmse_max=rmse_max, nbins=100, plottype=plottype)  
            axs[i,j].text(0.1, 0.9, f'{uselabels[i]} band', fontsize=10, transform=axs[i,j].transAxes, weight='bold')
            axs[i,j].invert_xaxis(); axs[i,j].invert_yaxis() 
        axs[0,j].set_title(f'{modelnames[models[j]]}', pad=5, fontsize=14)
        if sample_method != None and len(preds)!=2: axs[j,0].text(0.05, 0.6, 'NOTE: Point-predicting model: \npredictions are not samples \nfrom predicted \ndistributions', fontsize=8, transform=axs[0,j].transAxes)
    fig.supxlabel('Hydro Truth [mag]', fontsize=16)
    fig.supylabel(f'Model Prediction [mag]', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def compare_phot_err(taskdir, models, modelnames, sample_method=None, N=100, uselabels=None):
    '''
    For given models, compare prediction error vs true values for each phot target
    NOTE: NEED TO ADD THE FOLLOWING FUNCTIONALITY:
        - If 'plot_dist' is False
                A single point sample is drawn from the guassian for each observation
                The plotted bounds represent the distribution of errors for galaxies within a binned range of true values
        - If 'plot_dist' is True
                A large number of samples are drawn from the guassian for each observation
                The plotted bounds represent the distribution of errors for galaxies with a given true value
    '''

    if uselabels == None: # if not specified, use all labels (and get from any model's config - currently assuming all predict same targets)
        configfile = f'{taskdir}/FHO/{models[0]}_config.json' if 'FHO' in models[0] else f'{taskdir}/{models[0]}/expfile.json'
        uselabels = json.load(open(configfile, 'rb'))['data_params']['use_targs']
    fig, axs = plt.subplots(len(uselabels), len(models), figsize=(4*len(models), len(uselabels)*2), sharex=True, sharey=True)
    for j in range(len(models)):
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method, N) # returns [n_targs, n_obs] if sample_method is None, [n_targs, n_obs, n_samples] if sample_method is not None
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        for i in range(len(uselabels)):
            yhat = yhats[:,i] 
            y = ys[:,i]   
            rel_err = 100*(yhat - y)/y
            axs[i,j] = plot_binned_dist(axs[i,j], y, rel_err,  nstatbins=25)  
            axs[i,j].axhline(0, color='black', linestyle='--')
            axs[i,0].text(0.1, 0.86, f'{uselabels[i]} band', fontsize=12, transform=axs[i,j].transAxes, weight='bold')
            rho = stats.pearsonr(y, yhat).statistic # = np.cov(np.stack((y, yhat, axis=0))/(np.std(y)*np.std(yhat)) # pearson r
            bias = np.mean(y-yhat)  # mean of the residuals
            scatter = np.std(y-yhat) # SD of the residuals
            axs[i,j].text(0.75, 0.20, f'r: ' + "{:.3f}".format(np.round(rho,3)), fontsize=10, transform=axs[i,j].transAxes,)
            axs[i,j].text(0.75, 0.12, r'$\sigma$: ' + "{:.3f}".format(np.round(scatter,3)), fontsize=10, transform=axs[i,j].transAxes)
            axs[i,j].text(0.75, 0.03, f'bias: ' +  "{:.3f}".format(np.round(bias,2)), fontsize=10, transform=axs[i,j].transAxes)
        axs[0,j].set_title(f'{modelnames[models[j]]}', fontsize=13)
        if sample_method != None and len(preds)!=2: axs[j,0].text(0.05, 0.6, 'NOTE: Point-predicting model: \npredictions are not samples \nfrom predicted \ndistributions', fontsize=8, transform=axs[0,j].transAxes)
    fig.supxlabel('Hydro Truth [Mag]', fontsize=16)
    fig.supylabel('Percent Error', fontsize=16)
    #note = 'sampled from predicted distributions' if sample else 'predicted means'
    plt.suptitle('Percent error for model predictions', fontsize=16) #{note}\n')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def compare_UVr(taskdir, dataset, models, modelnames, sample_method='mean', N=100, plottype='heat', slice_r=-21, t=''):
    '''
    For given models, compare predicted U-V color vs r magnitude
    NOTE: For a larger sample, using full TNG dataset for truth, not just test set, so positions are not directly comparable
    '''

    # Load and plot true
    all_data_combined = pd.read_pickle(open(f'../Data/vol100/{dataset}_all_data_combined.pkl','rb'))
    ys = np.vstack(all_data_combined['phot']) 
    Us = ys[:,0]; Vs = ys[:,2]; rs = ys[:,5]
    extent = [[-25, -14], [-3, 4]]# [[-25, -14], [-1, 1.5]] if sample_method == 'mean' else [[-25, -14], [-3, 4]] # approx limits of worst model (FHO3) when sampled - idk if this really captures things well though
    fig, axs = plt.subplots(1, len(models)+1, figsize=((len(models)+1)*3.5, 4), sharey=True, sharex=True)
    #trueUVr, x_pos, y_pos, im = axs[0].hist2d(rs, Us-Vs, bins=100, range=extent, cmap='Greys', norm=mpl.colors.LogNorm())
    axs[0].set(ylabel=f' U-V color', xlabel='r magnitude')
    axs[0].set_title('True', fontsize=10)  
    axs[0].invert_xaxis() 
    plot_inset_hist(axs[0], rs, Us-Vs, xloc=slice_r, placement='within', vline_range=extent[1], nbins=N, color='blue')
    
    # Plot predicted U-V color vs r magnitude
    for j in range(len(models)):
        # Get predicted values 
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method, N=N)
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        Uhats = yhats[:,0]; Vhats = yhats[:,2]; rhats = yhats[:,5] 
        # If multiple samples, plot 2D histogram with contours, if single sample, scatter color against mag
        if plottype == 'scatter':#sample_method == 'no_sample':
            sighats = preds[1]
            Usighats = sighats[:,0]; Vsighats = sighats[:,2]; rsighats = sighats[:,5]
            axs[j+1].plot(rhats, Uhats-Vhats, 'ko', markersize=2, alpha=0.1)
            UVerr = np.sqrt(Usighats**2 + Vsighats**2)  #np.sqrt(np.std(rsighats)**2 + np.std(Usighats-Vsighats)**2) # maybe not quite correct, since this is if the errors are independent
            axs[j+1].errorbar(rhats, Uhats-Vhats, xerr=rsighats, yerr=UVerr, color='k', ls='none', elinewidth=0.05, alpha=0.5)
            axs[j+1].legend(handles=[mpl.lines.Line2D([0], [0], color='k', lw=0.5, label='progogated U-V error'), mpl.lines.Line2D([0], [0], marker='o', markersize=5, markerfacecolor='k', color='w', label='sample')], loc='upper right')
        else:
            predUVr, x_pos, y_pos, im = axs[j+1].hist2d(rhats, Uhats-Vhats, bins=100, range=extent, cmap='Greys', norm=mpl.colors.LogNorm()) #predUVr, xedges, yedges = np.histogram2d(rhats, Uhats-Vhats, bins=100, range=extent, density=True) # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
            X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
            #flat_nonzero = predUVr.flatten()[predUVr.flatten() > 0]
            z_levels = 8 #[np.percentile(flat_nonzero,50)]# [np.percentile(flat_nonzero,10), np.percentile(flat_nonzero,50), np.percentile(flat_nonzero,75)] # np.percentile(vals.flatten(),2) = the bin val number for which 98% of the bins are above (98% of the data is within), since np.percentile(vals.flatten(),98) = the bin val number for which 98% of the data is below
            cs = axs[j+1].contour(X,Y, np.log(predUVr.T+1), levels=z_levels, linewidths=0.4, alpha=0.5, colors='white') # np.log(vals.T+1)
            # fmt = {l:s for l, s in zip(cs.levels, ['50%'])} # ['90%', '50%', '25%']
            # axs[j+1].clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
        # Report error between predicted and true distributions
        # predUVr = np.where(predUVr==0, 0.001, predUVr) # add small value to zero bins, otherwise KL_div will be inf for those bins # # COULD TRY instead normalizing only after histograming, and normalize to not exactly std normal to add some small values predUVr = (predUVr - np.mean(predUVr))/np.std(predUVr) 
        # kldiv = np.round(np.mean(special.kl_div(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same # relentr = np.round(np.mean(special.rel_entr(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same  # ks_stat, ks_Pval = np.round(stats.ks_2samp(trueUVr, predUVr), 2) # Tests H0: two dists are the same. High P_value = cannot reject H0
        # mmd = np.round(float(run_utils.MMD(trueUVr, predUVr, kernel='rbf')), 3) # Tests H0: two dists are the same. High P_value = cannot reject H0  
        # t = axs[j+1].text(0.15, 0.85, f'MMD: {mmd}', fontsize=10, transform=axs[j+1].transAxes)
        # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
        # Set ax labels and plot inset slice hist
        plot_inset_hist(axs[j+1], rhats, Uhats-Vhats, xloc=slice_r, placement='within', vline_range=extent[1], nbins=N, color='blue')
        axs[j+1].set_title(f'{modelnames[models[j]]}', fontsize=10)
        axs[j+1].set_xlabel(f'r magnitude')
        axs[j+1].invert_xaxis() 

    plt.suptitle(f'U-V color vs r magnitude{t}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig

def compare_CMDs(taskdir, dataset, models, modelnames, colors, sample_method='mean', N=100, plottype='KDE'):
    '''
    For given models, compare predicted color vs magnitude
    NOTE: For a larger sample, using full TNG dataset for truth, not just test set, so positions are not directly comparable
    NOTE: Contours appear to cut off because setting ax xlims and ylims to be just slightly larger than the extent of the true data
        Means that if the 3 sigma contours of the pred data extend past this range they will be cut off
        Could try making this more visualy apealing by have the largest sigma level be like 2 sigma or something 
        Or wait - but if thats the reason then I should never be cutting off a contour for the true data, but I am 
    '''

    extents = [] # extent = [[-25, -14], [-1, 1.5]] if sample_method == 'mean' else [[-25, -14], [-3, 4]] # approx limits of worst model (FHO3) when sampled - idk if this really captures things well though
    used_targs = ["U", "B", "V", "K", "g", "r", "i", "z"]  # hardcoding since haven't yet wanted to change ever
    slice_locs = {"B":-20,"V":-20, "K":-23, "r":-21, "i":-21,"z":-21}

    # Load true and plot color vs mag for all color sets
    all_data_combined = pd.read_pickle(open(f'../Data/vol100/{dataset}_all_data_combined.pkl','rb'))
    ys = np.vstack(all_data_combined['phot']) #  np.array(np.vstack(list(allphot.values())))
    trueCMDs = []
    fig, axs = plt.subplots(len(colors), len(models)+1, figsize=((len(models)+1)*4, len(colors)*3))#, sharey='row')
    for i in range(len(colors)):
        set = colors[i]
        c1_idx = used_targs.index(set[0]); c2_idx = used_targs.index(set[1])        
        C1s = ys[:,c1_idx]; C2s = ys[:,c2_idx]
        C3_name = set[1] if len(set) == 2 else set[2]
        C3s = ys[:,used_targs.index(C3_name)]
        extent =[[min(C3s)-0.5, max(C3s)+1], [min(C1s-C2s)-1*np.abs(max(C1s-C2s)-min(C1s-C2s)), max(C1s-C2s)+1*np.abs(max(C1s-C2s)-min(C1s-C2s))]] #extent =[[min(C3s)+1, max(C3s)-0.5], [min(C1s-C2s)+0.3*np.abs(max(C1s-C2s)-min(C1s-C2s)), max(C1s-C2s)-0.3*np.abs(max(C1s-C2s)-min(C1s-C2s))]]
        #print(f'{colors[i]}: {[[min(C3s), max(C3s)], [min(C1s-C2s), max(C1s-C2s)]]} --> {extent}')
        extents.append(extent)
        ax, trueCMD = plot_CMD(axs[i,0], C1s, C2s, C3s, extent, nbins=N, plottype=plottype)
        trueCMDs.append(trueCMD)
        axs[i,0].set_xlim(extents[i][0]); axs[i,0].set_ylim(extents[i][1])
        axs[i,0].set_ylabel(f' {set[0]}-{set[1]} color', fontsize=13); axs[i,0].set_xlabel(f'{set[-1]} magnitude', fontsize=13) 
        plot_inset_hist(axs[i,0], C3s, C1s-C2s, xloc=slice_locs[C3_name], placement='right', vline_range=extent[1], nbins=N, color='blue')
        axs[i,0].invert_xaxis() 
        axs[0,0].set_title('True', fontsize=15, pad=15) 
    
    # Plot predicted color vs mag for all color sets
    for j in range(len(models)):
        # Get predicted values 
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method, N)
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        for i in range(len(colors)):
            set = colors[i]
            c1_idx = used_targs.index(set[0]); c2_idx = used_targs.index(set[1])        
            C1s = yhats[:,c1_idx]; C2s = yhats[:,c2_idx]
            C3_name = set[1] if len(set) == 2 else set[2]
            C3s = yhats[:,used_targs.index(C3_name)]
            ax, predCMD = plot_CMD(axs[i,j+1],  C1s, C2s, C3s, extent, nbins=N, plottype=plottype)
            axs[i,j+1].set_xlim(extents[i][0]); axs[i,j+1].set_ylim(extents[i][1])
            # Report error between predicted and true distributions
            # predCMD = np.where(predUVr==0, 0.001, predCMD) # add small value to zero bins, otherwise KL_div will be inf for those bins # # COULD TRY instead normalizing only after histograming, and normalize to not exactly std normal to add some small values predUVr = (predUVr - np.mean(predUVr))/np.std(predUVr) 
            # kldiv = np.round(np.mean(special.kl_div(trueCMDs[i], predCMD)), 3) # mean over all bins. 0 if dists are exactly the same # relentr = np.round(np.mean(special.rel_entr(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same  # ks_stat, ks_Pval = np.round(stats.ks_2samp(trueUVr, predUVr), 2) # Tests H0: two dists are the same. High P_value = cannot reject H0
            # mmd = np.round(float(run_utils.MMD(trueUCMDs[i], predCMD, kernel='rbf')), 3) # Tests H0: two dists are the same. High P_value = cannot reject H0  
            # t = axs[i,j+1].text(0.15, 0.85, f'MMD: {mmd}', fontsize=10, transform=axs[i,j+1].transAxes)
            # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
            # Create inset axis with slice histogram, plot slice location
            plot_inset_hist(axs[i,j+1], C3s, C1s-C2s, xloc=slice_locs[C3_name], placement='right', vline_range=extents[i][1], nbins=N, color='blue')
            #plot_inset_hist(axs[i,j+1], C3s, C1s-C2s, xloc='marginal',placement='right', vline_range=extent[1], color='grey')
            # Set ax labels
            axs[i,j+1].set_xlabel(f'{set[-1]} magnitude', fontsize=13)
            axs[i,j+1].invert_xaxis() 
            axs[0,j+1].set_title(f'{modelnames[models[j]]}', fontsize=15, pad=15)

    for ax in axs.reshape(-1):
        ax.tick_params(axis='both',labelsize=8)

    #plt.suptitle('Color-Magnitude Distributions', fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return fig

def compare_UVr_slice(taskdir, dataset, models, modelnames, sample_method='mean'):
    '''
    For given models, compare predicted U-V color for galaxies in different r magnitude slices
    NOTE: For a larger sampled, using full TNG dataset for truth, not just test set, so positions are not directly comparable
    '''

    # Plotting constraints
    extent = [[-25, -14], [-1, 1.5]] if sample_method == 'mean' else [[-25, -14], [-4, 4]]
    nbins = 30 if sample_method == 'mean' else 100 
    slice_rs = [-21, -17]  #np.linspace(extent[0][0], extent[0][1], nslices+1) #slice_xs = np.linspace(0, nbins, nslices+1, dtype=int)
    fig, axs = plt.subplots(len(slice_rs), len(models)+1, figsize=((len(models)+1)*3, len(slice_rs)*2.5), sharey=True, sharex=True)

    # Load and plot true
    allphot = pd.read_pickle(open(f'../Data/vol100/{dataset}_allphot.pkl','rb'))
    ys = np.array(np.vstack(list(allphot.values())))
    Us = ys[:,0]; Vs = ys[:,2]; rs = ys[:,5]
    #trueUVr, xedges, yedges = np.histogram2d(rs, Us-Vs, bins=nbins, range=extent, density=True)
    for i in range(len(slice_rs)):
        idxs = np.where((rs > slice_rs[i]-0.5) & (rs < slice_rs[i]+0.5))[0]
        UV = Us[idxs] - Vs[idxs]  #UV = trueUVr[slice_xs[i],:]
        axs[i,0].hist(UV, bins=nbins, range=extent[1], color='grey', density=True)
        axs[i,0].semilogy()
        axs[i,0].set_ylabel(f'r mag {slice_rs[i]} +/- 0.5')
    axs[0,0].set_title('True', fontsize=10)
    
    # Plot predicted U-V color hist for each r magnitude slice
    for j in range(len(models)):
        if 'FHO' in models[j]:
            resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl'
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method)
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        Uhats = yhats[:,0]; Vhats = yhats[:,2]; rhats = yhats[:,5]
        # predUVr, xedges, yedges = np.histogram2d(rhats, Uhats-Vhats, bins=nbins, range=extent, density=True)
        for i in range(len(slice_rs)):
            idxs = np.where((rhats > slice_rs[i]-0.5) & (rhats < slice_rs[i]+0.5))[0]
            UVhat = Uhats[idxs] - Vhats[idxs]  #UV = predUVr[slice_xs[i],:]
            axs[i,j+1].hist(UVhat, bins=nbins, range=extent[1], color='grey', density=True)
            axs[i,j+1].semilogy()
        axs[0,j+1].set_title(f'{modelnames[models[j]]}', fontsize=10)

    fig.supylabel('U-V color')
    plt.suptitle('U-V color in r magnitude slices')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig

def compare_color_merghist(taskdir, dataset, models, modelnames, sample_method=None):
    '''
    For given models, compare predicted U-V color vs z_50
    '''

    example_model = [m for m in models if 'FHO' not in m][0] # just make sure not an FHO (totally could use, but then result path is different)
    ys, _, _ = pickle.load(open(f'{taskdir}/{example_model}/testres.pkl', 'rb')) # load from one model's result file - faster than looping over test set graphs
    U_true = ys[:,0]; V_true = ys[:,2]
    Vbins = [-22, -21, -20, -17, -13]
    fig, axs = plt.subplots(len(Vbins)-1, len(models)+1, figsize=((len(models)+1)*3, 1.5*len(Vbins)), sharey=True, sharex=True)

    for i in range(len(Vbins)-1):
        Utrue = U_true[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        Vtrue = V_true[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        MHmetrics = pickle.load(open(f'../Data/vol100/{dataset}_[0, 1, 3, 4]_[0, 1, 2, 3, 4, 5, 6, 7]_[70, 10, 20]_testMHmetrics.pkl', 'rb'))
        z_50 = np.array(MHmetrics['z_50'].values)
        z_50_true = z_50[np.where((V_true > Vbins[i]) & (V_true < Vbins[i+1]))[0]]
        axs[i,0] = plot_binned_dist(axs[i,0], z_50_true,  Utrue - Vtrue,  nstatbins = np.unique(z_50_true))
        axs[i,0].set_ylabel('U-V color')
        axs[-1,0].set_xlabel('z50')
    axs[0,0].set_title('Truth')

    for j in range(len(models)):
        if 'FHO' in models[j]:
            # resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl'
            # modelname = models[j]
            raise NameError('Cant yet add FHOs because cant go back and create MH metrics since xs are not full graphs. And cant use the same as for the GNN models becasue used random TTS. If need to do this, will need to at least re-test the FHOs on final halos data created from the same graphs in the GNN test set.')
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
            MHmetrics = pickle.load(open(f'../Data/vol100/{dataset}_[0, 1, 3, 4]_[0, 1, 2, 3, 4, 5, 6, 7]_[70, 10, 20]_testMHmetrics.pkl', 'rb'))
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method) 
        Uhats = yhats[:,0]; Vhats = yhats[:,2]  
        for i in range(len(Vbins)-1):
            Uhat = Uhats[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            Vhat = Vhats[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            z_50 = np.array(MHmetrics['z_50'].values)
            z_50_pred = z_50[np.where((Vhats > Vbins[i]) & (Vhats < Vbins[i+1]))[0]]
            axs[i,j+1] = plot_binned_dist(axs[i,j+1], z_50_pred,  Uhat - Vhat,  nstatbins = np.unique(z_50_pred))
            axs[0,j+1].set_title(modelnames[models[j]])
            axs[-1,j+1].set_xlabel('z50')
            axs[i,int((len(models)+1)/2)].text(0.75, 0.85, f'V mag bin {Vbins[i]} to {Vbins[i+1]} ', transform=axs[i,1].transAxes, backgroundcolor='w')
        
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig   

def compare_corr_band_old(taskdir, vol, dataset, models, modelnames, by='V mag', sample_method=None, plotdiff=False, cutoff=None):
    '''
    For given models, compare predicted PS for each band OR predicted-true PS for each band
    '''

    # Load test data z0xall to get positions [should apply to all models using same dataset]
    mock_data_params = {'data_path': f"../Data/{vol}",                                # vol for these models 
                        'data_file': f"{dataset}_phot_final.pkl",                     # dataset for these models
                        'splits': [70, 10, 20], "transf_name": "QuantileTransformer", # hardcoding since haven't yet wanted to change ever
                        'use_feats': ["Mvir(10)"],                                    # doesn't matter (just need one that exists) since just using saved z0xall
                        'use_targs': ["U", "B", "V", "K", "g", "r", "i", "z"]}        # hardcoding since haven't yet wanted to change ever
    testdata, test_z0xall = pickle.load(open(data_utils.get_subset_path(mock_data_params, set='test'), 'rb')) # testdata, test_z0xall = pickle.load(open(f'{taskdir}/exp3/testdata.pkl', 'rb')) # get test ys from any model (all have same test set)
    z0xall = np.vstack(test_z0xall)
    _, tree_meta, _, _ = data_utils.read_data_meta(f"../Data/{vol}/{dataset}_meta.json", "phot")
    all_featnames = tree_meta['featnames']
    pos = np.vstack([z0xall[:, all_featnames.index('x(17)')], z0xall[:, all_featnames.index('y(18)')], z0xall[:, all_featnames.index('z(19)')]]).T # Mpc/h
    
    # Load test data other targs to get Mstar
    all_otherSHtargs = pickle.load(open(f'../Data/{vol}/{dataset}_subfind_othertargs.pkl', 'rb'))
    testidxs = np.load(osp.expanduser(f"../Data/{vol}/{dataset}_testidx20.npy")) 
    all_testtargs =  pd.DataFrame(columns = all_otherSHtargs.columns)
    for i in range(len(all_otherSHtargs)):
        if i in testidxs: all_testtargs.loc[len(all_testtargs)] = all_otherSHtargs.iloc[i]
    mstar = np.log10(np.array(all_testtargs['SubhaloStelMass'], dtype=float)*(10**10)/0.704) # mstar for test set (e.g. transformed)

    # Load test data truth values [load from one model's result file - faster than looping over test set graphs]
    example_model = [m for m in models if 'FHO' not in m][0] # just make sure not an FHO (totally could use, but then result path is different)
    ys, _, _ = pickle.load(open(f'{taskdir}/{example_model}/testres.pkl', 'rb')) 
    V_true = ys[:,2]; U_true = ys[:,0]

    # Plotting constants
    nc = 128
    bs = 110.7*0.704 # want in units of Mpc/h
    if by == 'V mag': bins = [-22, -21, -20, -18, -15, -13]
    if by == 'U-V': raise NotImplementedError('Set bins U-V!')
    colors = cm.get_cmap(truncate_colormap(cm.get_cmap('jet'), 0.3, 0.7))(np.linspace(0,1,len(bins)-1)) # pl.cm.jet(np.linspace(0,1,len(Vbins)-1))
    
    # If not plotting difference, plot for true bins
    if plotdiff:
        fig, axs = plt.subplots(1, len(models), figsize=((len(models))*5, 4), sharey=True, sharex=True)
    else:
        fig, axs = plt.subplots(1, len(models)+1, figsize=((len(models)+1)*5, 4), sharey=True, sharex=True)
        for i in range(len(bins)-1):
            if by == 'V mag':
                idxs = np.where(V_true < bins[i+1])[0] # np.where((val > bins[i]) & (val < bins[i+1]))[0]
            elif by == 'U-V':
                idxs = np.where((V_true-U_true  > bins[i]) & (V_true-U_true  < bins[i+1]))[0]
            pos_true_bin = pos[idxs]
            pos_true_bin_nc = (pos_bin/bs) * nc # want in units of nc
            mstar_bin = mstar[idxs]  
            field_true = paintcic(pos_true_bin_nc, mass=mstar_bin, bs=bs, nc=nc)
            field_true = field_true/np.mean(field_true)
            k_true, p_true = power(field_true, boxsize=bs)
            p_true_scl = p_true - sum(mstar_bin)/bs**3 # sum of all masses in bin divided by volume
            axs[0].plot(k_true, p_true, c=colors[i])
        axs[0].set_xlabel(r'k [$h^{-1} Mpc$]')
        axs[0].set_title(f'Binned by Truth {by}', fontsize=10)
        if cutoff != None: axs[0].set_ylim(-50, cutoff)

    # Plot for pred bins
    for j in range(len(models)):
        if 'FHO' in models[j]:
            resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl'
            model_ds = json.load(open(f"{taskdir}/FHO/{models[j]}_config.json","rb"))['data_params']['data_file'].replace('.pkl', '')
            #raise NameError('Cant yet add FHOs because use different test set, so pos and mstar dont correspond. If need to do this, will need to at least re-test the FHOs on final halos data created from the same graphs in the GNN test set.')
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
            model_ds = json.load(open(f"{taskdir}/{models[j]}/expfile.json","rb"))['data_params']['data_file'].replace('.pkl', '')
        if dataset not in model_ds:
                print(f'WARNING: {models[j]} was trained on {model_ds} not {dataset}. Shouldnt compare models run on different datasets')
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats  = data_utils.get_yhats(preds, sample_method) 
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
                rel_err = 100*(p_pred_scl-p_true_scl)/p_true_scl
                label = f'{by} {bins[i]} to {bins[i+1]}' if by == 'U-V' else f'{by} < {bins[i+1]}'
                axs[ax_idx].plot(k_pred, rel_err, label=label, c=colors[i])   
            else:
                axs[ax_idx].plot(k_pred, p_pred_scl, label=f'{by} {bins[i]} to {bins[i+1]}', c=colors[i])  
            if cutoff != None: axs[ax_idx].set_ylim(-50, cutoff) 
            axs[ax_idx].set_xlabel(r'k [$h^{-1} Mpc$]')
            axs[ax_idx].set_title(f"Bined {by} from Predictions of {modelnames[models[j]]}", fontsize=10) # f"bined by {by} sampled from {modelname} predictions"

    axs[-1].legend()
    xlab = r'percent error' if plotdiff else r'p(k) [$(h^{-1} Mpc)^3$]' 
    axs[0].set_ylabel(xlab)
    #plt.semilogx(); plt.semilogy()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(top=0.85)
    t = f'Error in p(k) between galaxies binned by true {by} and binned by predicted {by}' if plotdiff else f'Power spectra binned by predicted {by}'
    plt.suptitle(f'{t}', fontsize=12)

    return fig

def compare_corr_band(taskdir, vol, dataset, models, modelnames, modelcolors, by='V mag', sample_method=None, N=1000, plotdiff=True, bin_by='within'):
    '''
    For given models, compare predicted-true PS for each band 
    Same as above, but seperate ax for each band, not each model
    NOTE: only contains functionality for distribution-computing models
    NOTE: per conversation with Chirag, need more galaxies
        Probabaly need to use train + test set
        In which case need to load in info for all positions (df = pickle.load(open('explore_prep_MvirCut10_lenCut20000_resCut100_r50Cut0.7.pkl', 'rb')))
        And also need to re-test using the entire train + test set to get predictions on all
    '''

    # Load test data z0xall to get positions [should apply to all models using same dataset]
    mock_data_params = {'data_path': f"../Data/{vol}",                                # vol for these models 
                        'data_file': f"{dataset}_phot_final.pkl",                     # dataset for these models
                        'splits': [70, 10, 20], "transf_name": "QuantileTransformer", # hardcoding since haven't yet wanted to change ever
                        'use_feats': ["Mvir(10)"],                                    # doesn't matter (just need one that exists) since just using saved z0xall
                        'use_targs': ["U", "B", "V", "K", "g", "r", "i", "z"]}        # hardcoding since haven't yet wanted to change ever
    _, test_z0xall = pickle.load(open(data_utils.get_subset_path(mock_data_params, set='test'), 'rb')) # testdata, test_z0xall = pickle.load(open(f'{taskdir}/exp3/testdata.pkl', 'rb')) # get test ys from any model (all have same test set)
    z0xall = np.vstack(test_z0xall)
    _, tree_meta, _, _ = data_utils.read_data_meta(f"../Data/{vol}/{dataset}_meta.json", "phot")
    all_featnames = tree_meta['featnames']
    pos = np.vstack([z0xall[:, all_featnames.index('x(17)')], z0xall[:, all_featnames.index('y(18)')], z0xall[:, all_featnames.index('z(19)')]]).T # Mpc/h
    
    # PS constants
    nc = 128
    bs = 110.7*0.704 # want in units of Mpc/h
    bins = [-22, -21, -20, -18, -15, -13] # true goes from -25.376022 (brightest) to -14.138756 (dimmest)
    
    # Get truth PS for each V-mag bin [load from one model's result file - faster than looping over test set graphs]
    example_model = [m for m in models if 'FHO' not in m][0] # just make sure not an FHO (totally could use, but then result path is different)
    ys, _, _ = pickle.load(open(f'Task_phot/{example_model}/testres.pkl', 'rb')) 
    V_true = ys[:,2]
    ps_true_allbins = [] # PS for each bin
    N_true_allbins = [] # N gals in each bin when binning by true
    for i in range(len(bins)-1):
        if bin_by == 'within': idxs = np.where((V_true > bins[i]) & (V_true < bins[i+1]))[0] 
        if bin_by == 'brighter': idxs = np.where(V_true < bins[i+1])[0]
        pos_true_bin = pos[idxs]
        k, p_true = get_PS(pos_true_bin, bs, nc)
        ps_true_allbins.append(p_true)
        N_true_allbins.append(len(idxs))

    # If not plotting difference, plot the truth PS for each bin
    fig, axs = plt.subplots(len(bins)-1, 1, figsize=(6, (len(bins)-1)*4), sharex=True)
    if not plotdiff:
        for i in range(len(bins)-1):
            ps_bin = np.array(ps_true_allbins[i])
            axs[i].plot(k, ps_bin, c='black', label=f'True ({N_true_allbins[i]} galaxies)')  # k should be the same for all models
            axs[i].set_ylabel(r'p(k) [$(h^{-1} Mpc)^3$]')
            binlabel = f'{bins[i]} < V mag < {bins[i+1]}' if bin_by == 'within' else f'V mag < {bins[i+1]}'
            axs[i].text(0.4, 0.9, binlabel, transform=axs[i].transAxes)
        
    # Loop over models
    for j in range(len(models)):
        
        # Get the predicted PS for each V-mag bin, for each realization
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method, N) 
        V_preds = yhats[:,:,2] # not flattened - 10000 realizations per obs
        ps_preds_allbins, rel_errs_allbins, N_pred_allbins = [], [], [] # PS, rel err on PS and N gals for each realization, for each bin
        for i in range(len(bins)-1):
            ps_preds_bin, rel_errs_bin, N_pred_bin = [], [], [] # PS, rel err on PS, and N gals in each bin for each realization
            for k in range(N):
                # Get PS in this bin for this realization
                V_pred = V_preds[k]
                if bin_by == 'within': idxs = np.where((V_pred > bins[i]) & (V_pred < bins[i+1]))[0] # indxs of galaxies with predicted V within bin 
                if bin_by == 'brighter': idxs = np.where(V_pred < bins[i+1])[0] 
                pos_bin = pos[idxs]
                k, p_pred = get_PS(pos_bin, bs, nc)
                # Get rel err with true PS in this bin
                p_true = ps_true_allbins[i]
                rel_err = 100*(p_pred-p_true)/p_true
                # Add to lists
                ps_preds_bin.append(p_pred); rel_errs_bin.append(rel_err)
                N_pred_bin.append(len(idxs))
            ps_preds_allbins.append(ps_preds_bin)
            rel_errs_allbins.append(rel_errs_bin)
            N_pred_allbins.append(N_pred_bin)

        # Plot the mean and sd of the rel err for each bin, or the raw PS
        for i in range(len(bins)-1):
            val_bin = np.array(rel_errs_allbins[i]) if plotdiff else np.array(ps_preds_allbins[i])
            mean = np.nanmean(val_bin, axis=0) # mean over all realizations in this bin for each observations
            sd = np.nanstd(val_bin, axis=0) # sd over all realizations in this bin for all observations
            mean_N_gals = int(np.nanmean(N_pred_allbins[i])) # mean over all realizations in this bin for all observations
            sd_N_gals = int(np.nanstd(N_pred_allbins[i])) # sd over all realizations in this bin for all observations
            axs[i].plot(k, mean, c=modelcolors[models[j]], linestyle='-.', label=rf'{modelnames[models[j]]} ({mean_N_gals}$\pm${sd_N_gals} galaxies)')  # k should be the same for all models, since pos is the same
            axs[i].fill_between(k, mean-sd, mean+sd, color=modelcolors[models[j]], alpha=0.2)
            if not plotdiff: axs[i].semilogy()
            axs[i].set_ylabel('percent error' if plotdiff else r'p(k) [$(h^{-1} Mpc)^3$]')
            binlabel = f'{bins[i]} < V mag < {bins[i+1]}' if bin_by == 'within' else f'V mag < {bins[i+1]}'
            axs[i].text(0.4, 0.9, binlabel, transform=axs[i].transAxes)
            axs[i].legend(fontsize=8, loc='lower right', edgecolor='white')

    axs[-1].set_xlabel(r'k [$h^{-1} Mpc$]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f'Error in p(k) between galaxies binned by\ntrue V mag and vs predicted V mag' if plotdiff else 'Power spectra binned by predicted V mag')

    return fig

def compare_light_mass(taskdir, vol, dataset, models, modelnames, sample_method=None):
    '''
    For given models, compare predicted light to mass ratios
    - Use K band magnitude (Rachel sugests L-band, but TNG doesnt provide that -  what motivates choice of band here?)
    '''

    # Load test data z0xall to get Mvir [should apply to all models using same dataset]
    mock_data_params = {'data_path': f"../Data/{vol}",                                # vol for these models 
                        'data_file': f"{dataset}_phot_final.pkl",                     # dataset for these models
                        'splits': [70, 10, 20], "transf_name": "QuantileTransformer", # hardcoding since haven't yet wanted to change ever
                        'use_feats': ["Mvir(10)"],                                    # doesn't matter (just need one that exists) since all have Mvir
                        'use_targs': ["U", "B", "V", "K", "g", "r", "i", "z"]}        # hardcoding since haven't yet wanted to change ever
    testdata, test_z0xall = pickle.load(open(data_utils.get_subset_path(mock_data_params, set='test'), 'rb')) # testdata, test_z0xall = pickle.load(open(f'{taskdir}/exp3/testdata.pkl', 'rb')) # get test ys from any model (all have same test set)
    z0xall = np.vstack(test_z0xall)
    _, tree_meta, _, _ = data_utils.read_data_meta(f"../Data/{vol}/{dataset}_meta.json", "phot")
    Mvir = z0xall[:, tree_meta['featnames'].index('Mvir(10)')]

    # Load test data truth values [load from one model's result file - faster than looping over test set graphs]
    example_model = [m for m in models if 'FHO' not in m][0] # just make sure not an FHO (totally could use, but then result path is different)
    ys, _, _ = pickle.load(open(f'{taskdir}/{example_model}/testres.pkl', 'rb')) # load from one model's result file - faster than looping over test set graphs
    Ks = ys[:,3]

    # Plot
    fig, axs = plt.subplots(1, len(models)+1, figsize=((len(models)+1)*3, 4), sharey=True)
    #extent = [[-23, -14], [-4, 4]] # approx limits of worst model (FHO3) when sampled - idk if this really captures things well though
    trueMLR, xedges, yedges = np.histogram2d(Mvir, Ks, bins=100, density=True) #range=extent,  # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
    axs[0].plot(Mvir, Ks, 'ko', markersize=2, alpha=0.1)
    axs[0].set(ylabel=r'M$_{vir}$', xlabel='K magnitude')
    axs[0].set_title('True', fontsize=10)
    for j in range(len(models)):
        if 'FHO' in models[j]:
            resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl'
        else: 
            resfile = f'{taskdir}/{models[j]}/testres.pkl'
        _, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = data_utils.get_yhats(preds, sample_method)
        Khats = yhats[:,3]
        predMLR, xedges, yedges = np.histogram2d(Mvir, Khats, bins=100, density=True) #range=extent, # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
        preMLR = np.where(predMLR==0, 0.001, predMLR) # add small value to zero bins, otherwise KL_div will be inf for those bins # # COULD TRY instead normalizing only after histograming, and normalize to not exactly std normal to add some small values predUVr = (predUVr - np.mean(predUVr))/np.std(predUVr) 
        kldiv = np.round(np.mean(special.kl_div(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same
        # relentr = np.round(np.mean(special.rel_entr(trueUVr, predUVr)), 3) # mean over all bins. 0 if dists are exactly the same
        # ks_stat, ks_Pval = np.round(stats.ks_2samp(trueUVr, predUVr), 2) # Tests 
        # H0: two dists are the same. High P_value = cannot reject H0
        mmd = np.round(float(run_utils.MMD(trueUVr, predUVr, kernel='rbf')), 3) # Tests H0: two dists are the same. High P_value = cannot reject H0  
        t = axs[j+1].text(0.6, 0.15, f'KLD: {kldiv}\nMMD: {mmd}', fontsize=10, transform=axs[j+1].transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
        axs[j+1].plot(rhats, Uhats-Vhats, 'ko', markersize=2, alpha=0.1)
        axs[j+1].set_title(f'{modelnames[models[j]]}', fontsize=10)
        axs[j+1].set_xlabel(f'K magnitude')
        if sample_method != None and len(preds)==2: axs[j+1].text(0.5, 0.8, 'NOTE: data sampled from \npredicted distributions', fontsize=6, transform=axs[j+1].transAxes)
    note = 'sampled from predicted distributions' if sample_method != None else 'using predicted means'
    plt.suptitle(f'U-V color vs r magnitude {note}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig


def compare_props_preds(taskdir, models, modelnames, labelunits, sample_method=None, uselabels=None):
    '''
    For given models, compare predicted vs true values for each props target
    '''
    
    #l1, l2 = -25, -12
    # rmse_min = 0.5; rmse_max = 1.3
    # if sample_method != None: l2 += 1
    if uselabels == None: # if not specified, use all labels (and get from any model's config - currently assuming all predict same targets)
        configfile = f'{taskdir}/FHO/{models[0]}_config.json' if 'FHO' in models[0] else f'{taskdir}/{models[0]}/expfile.json'
        uselabels = json.load(open(configfile, 'rb'))['data_params']['use_targs']
    fig, axs = plt.subplots(len(uselabels), len(models), figsize=(4*len(models), len(uselabels)*2.5), sharey='row')
    for j in range(len(models)):
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats  = data_utils.get_yhats(preds, sample_method)
        sighats = preds[1]
        if yhats.ndim == 3:
            yhats, ys = data_utils.flatten_sampled_yhats(yhats, ys)
        for i in range(len(uselabels)):
            plot_targ(axs[i,j], yhats[:,i], sighats[i,:], ys[:,i])  
            axs[i,0].set_ylabel(f'{labelunits[uselabels[i]]}')
        axs[0,j].set_title(f'{modelnames[models[j]]}', fontsize=10)
    fig.supylabel('Model Prediction')
    fig.supxlabel('Hydro Truth')
    plt.tight_layout()

    return fig

def compare_sigma_hists(taskdir, models, modelnames, uselabels, plot_zscores=True):
    '''
    For given models, look at the distribution of the predicted sigmas (and the distirbution of z-scores if desired) 
    NOTE: If using a guassian loss func, and sigmas not roughly guassian, something is going wrong
          Not necesarily true if using Navarro since not constraining to be guassian 
    '''

    n_rows = 2 if plot_zscores else len(uselabels)
    fig, axs = plt.subplots(n_rows, len(models), figsize=(3*len(models), n_rows*2.5))# sharey=True)
    for j in range(len(models)):
        resfile = f'{taskdir}/FHO/{models[j]}_testres.pkl' if 'FHO' in models[j] else f'{taskdir}/{models[j]}/testres.pkl'
        ys, preds, _ = pickle.load(open(resfile, 'rb'))
        yhats = preds[0]; sighats = preds[1]
        for i in range(len(uselabels)):
            y = ys[:,i] 
            yhat = yhats[:,i] 
            sighat = sighats[:,i] 
            if plot_zscores:
                z = (y - yhat)/sighat
                axs[0,j].hist(sighat, bins=50, alpha=0.5, label=uselabels[i]); axs[0,j].set_xlabel('Predicted sig [mag]') # density=True, 
                #axs[0,j].semilogy()
                axs[0,j].legend(loc='upper left')
                axs[0,j].tick_params(axis='both',labelsize=8); axs[0,j].yaxis.offsetText.set_fontsize(8); axs[0,j].xaxis.offsetText.set_fontsize(8)  # need the later to adjust the offset sci not indcator size
                axs[1,j].hist(z, bins=50, alpha=0.5, density=True, label=uselabels[i]); axs[1,j].set_xlabel('z-score')
                axs[1,j].plot(np.linspace(min(z)-0.5, max(z)+0.5), stats.norm.pdf(np.linspace(min(z)-0.5, max(z)+0.5), 0, 1), linestyle='dashed', color='grey') # unit guassian for comparison
                axs[1,j].tick_params(axis='both',labelsize=8); axs[1,j].yaxis.offsetText.set_fontsize(8); axs[1,j].xaxis.offsetText.set_fontsize(8) 
                axs[1,j].legend(loc='upper left')
            else:
                axs[i,j].hist(sighat, bins=50, alpha=0.5) # density=True                  
                axs[i,j].text(0.1, 0.9, f'{uselabels[i]} band', fontsize=8, transform=axs[i,j].transAxes, weight='bold')
                axs[i,j].text(0.6, 0.25, r'Mean $\hat{\sigma}$ '+str(np.round(np.mean(sighat),2))+'\n'+r'SD $\hat{\sigma}$ '+str(np.round(np.std(sighat),3)), fontsize=8, transform=axs[i,j].transAxes)
        axs[0,j].set_title(f'{modelnames[models[j]]}', fontsize=10)
        if plot_zscores == False: fig.supxlabel('Predicted sig [mag]', fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.4) if plot_zscores else plt.subplots_adjust(wspace=0, hspace=0)

    return fig

###########################
# Plotting helper functions
###########################

def plot_inset_hist(ax, xs, ys, xloc, placement, vline_range, nbins, color='blue'):  
    '''
    Add insed marginal or slice hist for 2D distribution plot
    '''

    if placement == 'within':
        ins_ax = ax.inset_axes([.15, .1, .4, .25])   # [x, y, width, height] w.r.t. ax
        orientation = 'vertical'
        range =  None
        ins_ax.tick_params(axis='both',labelsize=8)
        ins_ax.semilogy()
    elif placement == 'right':
        ins_ax = ax.inset_axes([1, 0, 0.15, 1])
        orientation = 'horizontal'
        range = ax.get_ylim()
        ins_ax.set_xticks([]); ins_ax.set_yticks([])
        ins_ax.axis('off')
        ins_ax.semilogx()
    else: raise ValueError(f'placement must be "within" or "right" not {placement}')

    if xloc == 'marginal':
        vals = ys
    else:
        idxs = np.where((xs > xloc-0.5) & (xs < xloc+0.5))[0]
        vals = ys[idxs]
        ax.vlines(xloc, vline_range[0], vline_range[1], color=color, linestyle='--', alpha=0.5)

    ins_ax.hist(vals, bins=nbins, color=color, alpha=0.5, range=range, density=True, orientation=orientation) 
    

    return ax

def plot_binned_dist(ax, x, y, nstatbins): 
    '''
    Bin 2d distribution by x value and plot on given ax
    '''

    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=nstatbins)
    bin_99th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 99), bins=nstatbins)
    bin_1st, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 1), bins=nstatbins)
    bin_75th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 75), bins=nstatbins)
    bin_25th, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 25), bins=nstatbins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    ax.plot(bin_centers[np.isfinite(bin_means)], bin_means[np.isfinite(bin_means)], 'g')
    ax.fill_between(bin_centers, bin_25th, bin_75th, color='g', alpha=0.2, label='50%')
    ax.fill_between(bin_centers, bin_1st, bin_99th, color='g', alpha=0.07, label='98%')

    return ax

def plot_targ(ax, yhat, sighat, y, lims=None, textcmap=None, rmse_min=None, rmse_max=None, nbins=50, plottype='KDE'):
    '''
    Plot true vs predicted values on given axis
    '''

    # Plot pred vs true 
    if plottype == 'KDE':
        ax = az.plot_kde(y, yhat, ax=ax, hdi_probs=[0.393, 0.865, 0.989], contourf_kwargs={"cmap": "Blues"}, show=False)
    elif plottype == 'heat':
        vals, x_pos, y_pos, im = ax.hist2d(y, yhat, bins=nbins, range=[np.percentile(np.hstack([y,yhat]), [0,100]), np.percentile(np.hstack([y,yhat]), [0,100])], norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
        z_levels = 4 #[np.percentile(flat_nonzero,2), np.percentile(flat_nonzero,50)] # np.percentile(vals.flatten(),2) = the bin val number for which 98% of the bins are above (98% of the data is within), since np.percentile(vals.flatten(),98) = the bin val number for which 98% of the data is below
        cs = ax.contour(X,Y, vals.T, levels=z_levels, linewidths=0.4, alpha=0.8, colors='white') # np.log(vals.T+1)
    elif plottype == 'scatter':
        ax.plot(y, yhat, 'ko', markersize=1, alpha=0.05)
        ax.errorbar(y, yhat, yerr=sighat, color='k', ls='none', elinewidth=0.5, alpha=0.9)
        ax.legend(handles=[mpl.lines.Line2D([0], [0], color='k', lw=0.5, label='predicted sigma'), mpl.lines.Line2D([0], [0], marker='o', markersize=5, markerfacecolor='k', color='w', label='sample')], loc='upper right')
    else: raise ValueError(f'plottype must be "KDE", "heat" or "scatter" not {plottype}')
    
    # Report error 
    #rmse = np.round(float(np.std(yhat-y)),3)
    #rmse_color = textcmap(1-(rmse-rmse_min)/(rmse_max-rmse_min)) if textcmap is not None else 'black'
    # ax.text(0.6, 0.15, f'rmse: {rmse}', fontsize=10, transform=ax.transAxes, color=rmse_color)
    rho = stats.pearsonr(y, yhat).statistic # = np.cov(np.stack((y, yhat, axis=0))/(np.std(y)*np.std(yhat)) # pearson r
    rho_color = textcmap((rho-0.9)/(0.95-0.9)) if textcmap is not None else 'black'
    bias = np.mean(y-yhat)  # mean of the residuals
    scatter = np.std(y-yhat) # SD of the residuals
    ax.text(0.6, 0.22, f'r: ' + "{:.3f}".format(np.round(rho,3)), fontsize=13, transform=ax.transAxes, color=rho_color)
    ax.text(0.6, 0.15, r'$\sigma$: ' + "{:.3f}".format(np.round(scatter,3)), fontsize=13, transform=ax.transAxes)
    ax.text(0.6, 0.08, f'bias: ' +  "{:.3f}".format(np.round(bias,3)), fontsize=13, transform=ax.transAxes)
    
    # Set ax limits and plot perfect correspondence line
    if lims == None:
        lims = [(np.min(y)-0.5, np.max(y)+0.5), (np.min(yhat)-0.5, np.max(yhat)+0.5)]
    ax.plot([lims[0][0],lims[0][1]],[lims[1][0], lims[1][1]], 'k--', label='Perfect correspondance')
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.tick_params(axis='both', labelsize=8)

    return ax     

def plot_CMD(ax, C1s, C2s, C3s, extent, nbins=50, plottype='KDE'):
    '''
    Plot color (C1s - C2s) vs magnitude (C3s) on given axis
    Return 2D histogram for future comparison 
        Note that `extent` will set the limits on the returned 2D histogram, but not on the axis itself 
    '''

    if plottype == 'KDE':
        H, x_pos, y_pos = np.histogram2d(C3s, C1s-C2s, bins=nbins, range=extent)
        ax = az.plot_kde(C3s, C1s-C2s, ax=ax, hdi_probs=[0.393, 0.865, 0.989], contourf_kwargs={"cmap": "Greys"}, show=False)
    elif plottype == 'heat':
        H, x_pos, y_pos, im = ax.hist2d(C3s, C1s-C2s, bins=nbins, range=extent, cmap='Greys', norm=mpl.colors.LogNorm()) #predUVr, xedges, yedges = np.histogram2d(rhats, Uhats-Vhats, bins=100, range=extent, density=True) # set x and y ranges becasue otherwise kl div will be lower if predicted dist has huge range
        X, Y = np.meshgrid((x_pos[1:]+x_pos[:-1])/2, (y_pos[1:]+y_pos[:-1])/2)
        z_levels = 3 #[np.percentile(flat_nonzero,10), np.percentile(flat_nonzero,50), np.percentile(flat_nonzero,75)] # np.percentile(vals.flatten(),2) = the bin val number for which 98% of the bins are above (98% of the data is within), since np.percentile(vals.flatten(),98) = the bin val number for which 98% of the data is below
        cs = ax.contour(X,Y, H.T, levels=z_levels, linewidths=0.4, alpha=0.5, colors='white') # np.log(vals.T+1)
    elif plottype == 'scatter': 
        H, x_pos, y_pos = np.histogram2d(C3s, C1s-C2s, bins=nbins, range=extent)
        ax.plot(C3s, C1s-C2s, 'ko', markersize=1, alpha=0.1)
    else: raise ValueError(f'plottype must be "KDE", "heat" or "scatter" not {plottype}')

    return ax, H


##########################
# Graph plotting functions
##########################

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
        ms = []
        for q, key in enumerate(di.keys()):
            m = x[q][masscol] #feat = transformer[tkeys[k]].inverse_transform(graph.x.numpy()[q,k].reshape(-1, 1))[0][0]
            ms.append(m)
            di[key] = m
        nx.set_node_attributes(G, di, 'Mvir') # just a typo that he was calling this n_prog??
        labels = nx.get_node_attributes(G, 'Mvir') # dict of "index: value" pairs 
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
            ms = np.round(np.percentile(ms, np.linspace(0,100,9)),1)
            cbar = fig.colorbar(nc, shrink=0.5, ax=ax)
            cbar.ax.set_yticklabels(ms)
            cbar.set_label(r'Halo mass [log$_{10}$ $M_{\odot}$]') # cbar.set_label('Redshift')
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
# Clustering analysis functions
#   - 'power', 'ffthk', 'paintcic', and 'paint' and From Chirag Modi https://github.com/modichirag/contrastive_cosmology/
############################################

def get_PS(pos, bs, nc):

    #pos_nc = (pos/bs) * nc # want in units of nc
    field = paintcic(pos, bs=bs, nc=nc) # mass=mstar_true_bin
    field = field/np.mean(field)
    k, p = power(field, boxsize=bs)

    return k, p


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
        k = fftk(f1.shape, boxsize, symmetric=symmetric) # get wavenumber vector
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

