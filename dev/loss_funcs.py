from pytools import F
import torch
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, GaussianNLLLoss
from torch import log, sum, square
import sys
try: 
    from dev import run_utils
except ModuleNotFoundError:
    try:
        sys.path.insert(0, '~/ceph/ObsFromTrees_2024/ObservablesFromTrees/dev/')
        from ObservablesFromTrees.dev import run_utils
    except ModuleNotFoundError:
        sys.path.insert(0, 'ObservablesFromTrees/dev/')
        from dev import run_utils

#############################
###   Simple loss funcs  ####
#############################
def L2():
    return MSELoss()

def L1():
    return L1Loss()

def SmoothL1():
    return SmoothL1Loss(beta=0.5)

def MSE():
    return MSELoss()

def GaussNd_torch():
    return GaussianNLLLoss()

###################################
###  Negative log likelihoods  ####
###################################

def GaussNd(pred_mu, ys, pred_sig):
    '''
    Loss func to fit for sigma as a function of X and feature, assuming guassian errors.
     - pred_sig [n_targ, n_obs]: predicted SD of each target, e.g. diagonal of covariance matrix
    '''
    pred_var = pred_sig**2
    z = square(pred_mu-ys)/pred_var
    err_loss = sum(z)/2 
    sig_loss = sum(log(pred_var))
    loss = err_loss + sig_loss

    if torch.isnan(sig_loss):
        raise ValueError('Sigloss has become NaN')
    
    return loss, err_loss, sig_loss    


def Navarro(pred_mu, ys, pred_sig):
    '''
    From Navarro et al. 2020
    Loss func to fit for sigma as a function of X and feature, BUT allowing targets to have non-guassian errors (they can come from an arbitrary distribution). 
     - pred_sig [n_targ, n_obs]: predicted SD of each target, e.g. diagonal of covariance matrix
    '''
    pred_var = pred_sig**2
    mu_err = (pred_mu - ys)**2
    mu_loss = sum(log(sum(mu_err)))
    sig_err = ((pred_mu - ys)**2 - pred_var)**2
    sig_loss = sum(log(sum(sig_err)))
    loss = mu_loss + sig_loss

    return loss, mu_loss, sig_loss


def GaussNd_corr(pred_mu, ys, vec, device='cpu'):
    '''
    Loss func to fit for variances as a function of X and feature, assuming guassian errors and allowing covariance of features.
     - vec = unconstrained model output vector of shape [n_obs, n_targ*(n_targ-1)/2+n_targ]
    '''

    n_targ = len(ys[0,:])
    n_obs = len(ys)
    cov = run_utils.cov_from_vec(vec, n_targ, n_obs, device=device) 
    loss = - torch.distributions.MultivariateNormal(pred_mu, cov).log_prob(ys).sum() # mean() might be "better" but sum() works way better for FHOs, doesnt change GNN results much ...

    return loss 

