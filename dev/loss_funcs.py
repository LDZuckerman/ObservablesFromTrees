from pytools import F
import torch
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, GaussianNLLLoss
from torch import log, sum, square

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


def GuassNd_corr(pred_mu, ys, pred_fullsig):
    '''
    Loss func to fit for variances as a function of X and feature, assuming guassian errors and allowing covariance of features.
     - pred_cov [n_targ, n_targ, n_obs] (?): predicted covariance of each target
    '''

    # Steps
    #     Have network output vector “v”
    #     Reshape v to a lower triangular matrix L
    #           L = reshape(v) to get a lower triangular matrix. 
    #           Will need to write this. Could use something like https://stackoverflow.com/questions/73075280/how-do-i-convert-a-31-vector-into-a-22-lower-triangular-matrix-in-pytorch
    #     Get cov = L.L^T
    #     Compute loss = multivariate_normal(mean, cov).log_prob(x)

    # # chol = torch.cholesky(pred_cov) <- should instead predict the lower triangular so that dont need choslsky , then construct full cov from that. e.g. network predict entries of lower triangular (flattened). In loss, convert to lower triagnular, put 
    # # mu_err = (pred_mu - ys)**2
    # # y_dist = torch.distributions.MultivariateNormal(pred_mu, pred_cov)  
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
