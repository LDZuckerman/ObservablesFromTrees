from uu import Error
from pytools import F
import torch
from torch import log_
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch import log, sum, square, vstack, zeros, bmm, det, inverse
import numpy as np


###############################
###    Simple loss funcs   ####
###############################
def L2():
    return MSELoss()

def L1():
    return L1Loss()

def SmoothL1():
    return SmoothL1Loss(beta=0.5)

def MSE():
    return MSELoss()

###############################
###  Homemade loss funcs   ####
##########################################
###  All are negative log likelihoods   ####
########################################## 

def GaussNd(pred_mu, ys, pred_sig):
    '''
    Loss func to fit for sigma as a function of X and feature, assuming guassian errors.
     - pred_sig [n_targ, n_obs] (?): predicted std of each target, e.g. diagonal of covariance matrix
    '''
    z = square((pred_mu-ys)/pred_sig)
    err_loss = sum(z)/2 
    sig_loss = sum(log(pred_sig))

    if torch.isnan(sig_loss):
        raise ValueError('Sigloss has become NaN')
    
    return err_loss+sig_loss, err_loss, sig_loss    

# def GaussNd(pred, ys, var):
#     '''General uncorrelated gaussian'''
#     z = (pred-ys)/var 
#     sigloss = sum(log(var))
#     err_loss = sum((square(z)))/2 
    
#     return err_loss+sigloss, err_loss, sigloss   

def Navarro(pred_mu, ys, pred_sig):
    '''
    From Navarro et al. 2020
    Loss func to fit for sigma as a function of X and feature, BUT allowing targets to have non-guassian errors (they can come from an arbitrary distribution). 
     - pred_var [n_targ, n_obs] (?): predicted variance of each target, e.g. diagonal of covariance matrix
    '''

    mu_err = (pred_mu - ys)**2
    mu_loss = sum(log(sum(mu_err)))
    #pred_sig = torch.abs(pred_var)**0.5 #pred_sig = pred_var**0.5 # this becomes all nan because pred var contains negatives
    sig_err = ((pred_mu - ys)**2 - pred_sig**2)**2
    sig_loss = sum(log(sum(sig_err)))

    return mu_loss + sig_loss, mu_loss, sig_loss

def Navarro2(pred_mu, ys, pred_var):
    '''
    From Navarro et al. 2020
    Loss func to fit for sigma as a function of X and feature, BUT allowing targets to have non-guassian errors (they can come from an arbitrary distribution). 
     - pred_var [n_targ, n_obs] (?): predicted variance of each target, e.g. diagonal of covariance matrix
    '''

    mu_err = (pred_mu - ys)**2
    mu_loss = sum(log(sum(mu_err)))
    pred_sig = torch.abs(pred_var)**0.5 #pred_sig = pred_var**0.5 # this becomes all nan because pred var contains negatives
    sig_err = ((pred_mu - ys)**2 - pred_sig**2)**2
    sig_loss = sum(log(sum(sig_err)))

    return mu_loss + sig_loss, mu_loss, sig_loss

def GuassNd_corr(pred_mu, ys, pred_cov):
    '''
    Loss func to fit for sigma as a function of X and feature, assuming guassian errors and allowing covariance of features.
     - pred_cov [n_targ, n_targ, n_obs] (?): predicted covariance of each target
    '''

    # # chol = torch.cholesky(pred_cov)
    # # mu_err = (pred_mu - ys)**2
    # y_dist = torch.distributions.MultivariateNormal(pred_mu, pred_cov)  
    # log_prob = y_dist.log_prob(ys) # ??????
    
    raise NotImplementedError('Loss function not yet implemented')

 # If predicting exactly 2 targets?
def Gauss2d(pred, ys, var):
    '''2d Gaussian, can be subbed for GaussND'''

    sig1=var[:,0]
    sig2=var[:,1]
    z1=(pred[:,0]-ys[:,0])/sig1
    z2=(pred[:,1]-ys[:,1])/sig2
    sigloss=sum(log(sig1)+log(sig2))
    err_loss = sum((z1**2+z2**2)/2)
    
    return err_loss+sigloss, err_loss, sigloss
