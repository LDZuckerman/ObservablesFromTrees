from torch import sqrt, sum, square, no_grad, vstack, std, IntTensor
from torch.cuda import FloatTensor
import numpy as np

##################################################
### Could be defined in loop              ####
### different funcs to calculate metrics  ####
##################################################


def scatter(loader, model, n_targ):
    
    model.eval()
    correct = 0
    with no_grad():
        for dat in loader: 
            out = model(dat) 
            correct += sum(square(out - dat.y.view(-1,n_targ)))
    return sqrt(correct/len(loader.dataset)).cpu().detach().numpy()

def test_multi(loader, model, targs, l_func, scale):
    
    n_targ=len(targs)
    outs, ys = [], []
    model.eval()
    if scale:
        sca=FloatTensor(scales[targs])
        ms=FloatTensor(mus[targs])
    with no_grad(): 
        for data in loader: 
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr, Gauss4d_corr"]:
                out, var, rho = model(data) 
            ys.append(data.y.view(-1,n_targ))
            outs.append(out)
    outss = vstack(outs)
    yss = vstack(ys)
    return std(outss - yss, axis=0).cpu().detach().numpy(), yss.cpu().detach().numpy(), outss.cpu().detach().numpy()


def test_multi_varrho(loader, model, targs, l_func): 
    '''This one is the most updated'''

    n_targ=len(targs)
    outs, ys, vars, rhos = [], [], [], []
    model.eval()
    with no_grad(): 
        for data in loader: 
            rho = IntTensor(0)
            var = IntTensor(0)
            if l_func in ["L1", "L2", "SmoothL1"]: 
                out = model(data)  
            if l_func in ["Gauss1d", "Gauss2d", "GaussNd"]:
                out, var = model(data)  
            if l_func in ["Gauss2d_corr", "Gauss4d_corr"]:
                out, var, rho = model(data) 
            ys.append(data.y.view(-1,n_targ))
            outs.append(out)
            vars.append(var)
            rhos.append(rho)

    outss = vstack(outs) # preds
    yss = vstack(ys) # trues
    vars = vstack(vars) #
    rhos = vstack(rhos)

    return std(outss - yss, axis=0).cpu().detach().numpy(), yss.cpu().detach().numpy(), outss.cpu().detach().numpy(), vars, rhos

