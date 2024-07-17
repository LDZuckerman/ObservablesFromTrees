from torch import layer_norm
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Linear, LayerNorm, LeakyReLU, Module, ReLU, Sequential, ModuleList
from torch_geometric.nn import SAGEConv, global_mean_pool, norm, global_max_pool, global_add_pool, MetaLayer
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch import cat, square,zeros, clone, abs, sigmoid, float32, tanh, clamp, log
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



class Sage(Module):
    '''
    GNN model modified slighlty from implemnetation by Christian Jesperson https://github.com/astrockragh/Mangrove/blob/7c00646aeca47c484b5dae52ba6e4598c0367fe2/dev/models.py
    Node-only model built upon the GraphSAGE convolutional layer.
    '''

    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='ReLU', 
                    decode_layers=2, decode_activation='LeakyReLU', layernorm=True, get_sig=False, get_cov=False, agg='sum'): # get_rho=False, 
        super(Sage, self).__init__()

        self.encode=encode
        if self.encode:
            self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True) 
        self.decode_activation=decode_activation # keep this LeakyReLU
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=int(in_channels)
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.predict_sig=get_sig
        self.predict_cov=get_cov #get_rho
        self.agg=agg
        if self.predict_sig and self.predict_cov: raise ValueError('Must select either diag or full cov, not both')
        
        ########################
        # Convolutional Layers #
        ######################## 

        self.convs=ModuleList()
        if self.encode:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(int(conv_layers-1)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        #############
        # Mu Layers #
        #############    

        self.decoders = ModuleList()
        self.norms = ModuleList()
        for _ in range(out_channels):
            self.decoder=ModuleList()
            self.norm=ModuleList()
            for i in range(decode_layers):
                if i==decode_layers-1: ## if final layer, make layer with only one output
                    self.norm.append(LayerNorm(normalized_shape=hidden_channels))  
                    self.decoder.append(Linear(hidden_channels, 1))
                else:
                    self.norm.append(LayerNorm(normalized_shape=hidden_channels))
                    self.decoder.append(Linear(hidden_channels, hidden_channels))
            self.decoders.append(self.decoder)
            self.norms.append(self.norm)

        ######################################
        # SD Layers (identical to mu layers) #
        ######################################

        if self.predict_sig:
            self.sigs = ModuleList()
            self.sig_norms = ModuleList()
            for _ in range(out_channels):
                self.sig = ModuleList()
                self.sig_norm = ModuleList()
                for i in range(decode_layers):
                    if i == decode_layers-1:
                        self.sig_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.sig.append(Linear(hidden_channels, 1))
                    else:
                        self.sig_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.sig.append(Linear(hidden_channels, hidden_channels))
                self.sigs.append(self.sig)
                self.sig_norms.append(self.sig_norm)

        #############################################
        # Cov Layers (change shape of final output) #
        #############################################

        if self.predict_cov:
            self.covs = ModuleList()
            self.cov_norms = ModuleList()
            out_dim = int(out_channels*(out_channels-1)/2+out_channels)
            for _ in range(out_dim):
                self.cov_l = ModuleList()
                self.cov_norm = ModuleList()
                for i in range(decode_layers):
                    if i == decode_layers-1:
                        self.cov_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.cov_l.append(Linear(hidden_channels, 1))
                    else:
                        self.cov_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.cov_l.append(Linear(hidden_channels, hidden_channels))
                self.covs.append(self.cov_l)
                self.cov_norms.append(self.cov_norm)
        
        #####################
        # Activation Layers #
        #####################
        
        self.conv_act = run_utils.get_act_func(self.conv_activation) # self.conv_act_f()
        self.decode_act =  run_utils.get_act_func(self.decode_activation) # self.decode_act_f() ## could apply later


    def forward(self, graph):

        # Get data
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = x.float() # convert float64 to float32 

        # Apply simple MLP on x
        if self.encode:
            x = self.node_enc(x)
        
        # Apply graph convolutions to x and edges
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.conv_act(x) 
        if self.agg=='sum':  ## sum for physics
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        # Apply mu layers to x to get predicted mus
        mus=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1) 
            if torch.any(x1) > 100: raise ValueError('mu predictions have become very large')
            mus.append(x1)
        mus=cat(mus, dim=1)
        
        # Apply sig layers to x to get predicted sigs - same as above
        if self.predict_sig:
            logsig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1) 
                if torch.any(x1) > 100: raise ValueError('sig predictions have become very large')  
                logsig.append(x1)
            logsig=cat(logsig, dim=1) # Instead of taking abs, treat prediction as log sigmas

        # Apply cov layers to get predicted full cov - same as above 
        if self.predict_cov:
            cov=[]
            for norm, decode in zip(self.cov_norms, self.covs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1) 
                if torch.any(x1) > 100: raise ValueError('cov predictions have become very large') 
                cov.append(x1)
            cov=cat(cov, dim=1)

        #print(mus.shape, cov.shape)

        if self.predict_sig:
            return mus, logsig
        if self.predict_cov:
            return mus, cov
        else:
            return mus
    

############################
# Simple model for predicting targets from ONLY final halo properties
############################
        
class SimpleNet(nn.Module):

  def __init__(self, hidden_layers, in_channels, out_channels): 
    super(SimpleNet, self).__init__()
    self.linear1 = nn.Linear(in_channels, hidden_layers) 
    self.linear2 = nn.Linear(hidden_layers, out_channels) 
    
  def forward(self, x):
    x = torch.sigmoid(self.linear1(x))
    out = self.linear2(x)
    return out

############################
# Simple models for prediction from ONLY final halos
############################
  
class SimpleFullDistNet(nn.Module):
   '''
   Simple MLP that predicts mean and [  ] : 
        mu = mean of the target distribution
        logfullsig = [n_obs, n_targ*(n_targ-1)/2+n_targ]  e.g. (sqrt of the) flattened values in the lower triangular full covariance matrix of targets
   NOTE: does include LayerNorm
   NOTE: eventually should probabaly combine this with the below. Then just rerun the relevant MLPs for consistent naming
   '''
    
   def __init__(self, hyper_params, in_channels, out_channels): 
    super(SimpleFullDistNet, self).__init__()

    hidden_layers = hyper_params['hidden_layers']
    h = hyper_params['hidden_channels']
    act_func = run_utils.get_act_func(hyper_params['activation'])

    self.linears1 = nn.ModuleList([nn.Linear(in_channels, h), act_func]) 
    self.linears2 = nn.ModuleList([nn.Linear(in_channels, h), act_func])
    for i in range(hidden_layers):
        self.linears1.append(nn.Linear(h, h))
        self.linears1.append(act_func) 
        self.linears2.append(nn.Linear(h, h))
        self.linears2.append(act_func) 
    self.linears1.extend([nn.LayerNorm(h), nn.Linear(h, out_channels)]) 
    self.linears2.extend([nn.LayerNorm(h), nn.Linear(h, int(out_channels*(out_channels-1)/2+out_channels))]) # LT cov matrix has n_targ*(n_targ-1)/2 + n_targ non-zero entries
    
   def forward(self, x, debug=False):

    x1 = clone(x)
    for m in self.linears1:
        x1 = m(x1)
    mu = x1
    
    x2 = clone(x)
    for m in self.linears2:
        x2 = m(x2)
    logfullsig = x2
    
    if torch.stack([torch.isnan(p).any() for p in self.linears2.parameters()]).any(): raise ValueError('NaNs in weights of sigma layers')
    if torch.stack([torch.isnan(p).any() for p in self.linears1.parameters()]).any(): raise ValueError('NaNs in weights of mu layers')
    if torch.any(torch.abs(mu) > 100): raise ValueError('mu predictions have become very large')
    if torch.any(torch.abs(logfullsig) > 100): raise ValueError('logfullsig predictions have become very large')

    return mu, logfullsig
   
        
class SimpleDistNet(nn.Module):
   '''
   Simple MLP that predicts mean and SD of distribution of each target: 
        mu = mean of the target distribution
        logsig = SD of the target distribution e.g. (sqrt of the) variances on the diagonal of a full covariance matrix of targets
   NOTE: does include LayerNorm
   '''
    
   def __init__(self, hyper_params, in_channels, out_channels): 
    super(SimpleDistNet, self).__init__()

    hidden_layers = hyper_params['hidden_layers']
    h = hyper_params['hidden_channels']
    act_func = run_utils.get_act_func(hyper_params['activation'])

    print('  Model has hidden_layers', hidden_layers, 'h', h, 'in_channels', in_channels, 'out_channels', out_channels, 'act_func', act_func)

    self.linears1 = nn.ModuleList([nn.Linear(in_channels, h), act_func]) 
    self.linears2 = nn.ModuleList([nn.Linear(in_channels, h), act_func])
    for i in range(hidden_layers):
        self.linears1.append(nn.Linear(h, h))
        self.linears1.append(act_func) 
        self.linears2.append(nn.Linear(h, h))
        self.linears2.append(act_func) 
    self.linears1.extend([nn.LayerNorm(h), nn.Linear(h, out_channels)]) #self.linears1.append(nn.Linear(h, out_channels))
    self.linears2.extend([nn.LayerNorm(h), nn.Linear(h, out_channels)]) #self.linears2.append(nn.Linear(h, out_channels))
    # self.linears2.append(act_func()) # REMOVING see above comment
    
   def forward(self, x, debug=False):

    x1 = clone(x)
    for m in self.linears1:
        x1 = m(x1)
        if debug: print(f"{m} {x1[0,0:5].detach().numpy()}")
    mu = x1
    
    x2 = clone(x)
    for m in self.linears2:
        x2 = m(x2)
        if debug: print(f"{m} {x2[0,0:5].detach().numpy()}")
    logsig = x2
    
    if torch.stack([torch.isnan(p).any() for p in self.linears2.parameters()]).any(): raise ValueError('NaNs in weights of sigma layers')
    if torch.stack([torch.isnan(p).any() for p in self.linears1.parameters()]).any(): raise ValueError('NaNs in weights of mu layers')
    if torch.any(torch.abs(mu) > 100): raise ValueError('mu predictions have become very large. Mean(mu) =', torch.mean(mu).item())
    if torch.any(torch.abs(logsig) > 100): raise ValueError('logfullsig predictions have become very large. Mean(logsig) =', torch.mean(logsig).item())

    return mu, logsig
   
class SimpleDistNet2(nn.Module):
    '''
    A second of the above model; expansion of Chrisitan's pt estimate MLP model to include a second output for the standard deviation of the target
    Also, allow for hyperparameters to be passed in
    '''
    def __init__(self, hyper_params, in_channels, out_channels):
        super().__init__()
        hidden_layers = hyper_params['hidden_layers']
        h = hyper_params['hidden_channels']
        act_func = run_utils.get_act_func(hyper_params['activation'])
        layer_norm = eval(hyper_params['layer_norm'])

        layers = [Linear(in_channels, h), act_func]
        for i in range(hidden_layers):
            layers.append(Linear(h, h))
            layers.append(act_func) 
        if layer_norm:
            layers.append(LayerNorm(h))
        layers.append(Linear(h, out_channels))
        self.mlp = Sequential(*layers)

    def forward(self, x):

        mu = self.mlp(x)
        if torch.stack([torch.isnan(p).any() for p in self.mlp.parameters()]).any(): raise ValueError('NaNs in weights of mu layers')
        logsig = self.mlp(x) # just run the same mlp layers on the input twice, interpret the second output as the logsig
        if torch.stack([torch.isnan(p).any() for p in self.mlp.parameters()]).any(): raise ValueError('NaNs in weights of sigma layers')

        return mu, logsig

class MLP(Module):
    '''
    Chrisitan's MLP model
    Simple MLP class with ReLU activiation + layernorm
    '''
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super().__init__()
        layers = [Linear(n_in, hidden), ReLU()]
        for i in range(nlayers):
            layers.append(Linear(hidden, hidden))
            layers.append(ReLU()) 
        if layer_norm:
            layers.append(LayerNorm(hidden))
        layers.append(Linear(hidden, n_out))
        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)