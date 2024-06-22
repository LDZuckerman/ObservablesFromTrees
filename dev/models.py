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
    Model built upon the GraphSAGE convolutional layer. This is a node only model (no global, no edge).
    Model takes a data object from a dataloader in the forward call and takes out the rest itself. 
    '''

    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, get_sig=False, get_cov=False, agg='sum'): # get_rho=False, 
        super(Sage, self).__init__()

        self.encode=encode
        if self.encode:
            self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True) 
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=int(in_channels)
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.predict_sig=get_sig
        self.predict_cov=get_cov #get_rho
        self.agg=agg
        
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

        ########################
        # Mu ('decode') Layers #
        #########################     

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

        #################################
        # SD Layers (same as mu layers) #
        #################################

        if self.predict_sig:
            self.sigs = ModuleList()
            self.sig_norms = ModuleList()
            for _ in range(out_channels):
                self.sig=ModuleList()
                self.sig_norm=ModuleList()
                for i in range(decode_layers):
                    if i==decode_layers-1:
                        self.sig_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.sig.append(Linear(hidden_channels, 1))
                    else:
                        self.sig_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.sig.append(Linear(hidden_channels, hidden_channels))
                self.sigs.append(self.sig)
                self.sig_norms.append(self.sig_norm)

        ######################
        # Co-Variance Layers #
        ######################

        if self.predict_cov:
            self.rhos = ModuleList()
            self.rho_norms = ModuleList()
            for _ in range(self.rho):
                self.rho_l=ModuleList()
                self.rho_norm=ModuleList()
                for i in range(decode_layers):
                    if i==decode_layers-1:
                        self.rho_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.rho_l.append(Linear(hidden_channels, 1))
                    else:
                        self.rho_norm.append(LayerNorm(normalized_shape=hidden_channels))
                        self.rho_l.append(Linear(hidden_channels, hidden_channels))
                self.rhos.append(self.rho_l)
                self.rho_norms.append(self.rho_norm)
        
        #####################
        # Activation Layers #
        #####################
        
        self.conv_act=self.conv_act_f()
        self.decode_act=self.decode_act_f() ## could apply later

    def conv_act_f(self):
        if self.conv_activation =='relu':
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        # Get data
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = x.float() # convert float64 to float32 

        # Apply simple MLP on x
        if self.encode:
            x = self.node_enc(x)
        
        # PApply graph convolutions to x and edges
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.conv_act(x) 
        if self.agg=='sum':  ## sum for physics
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        # Apply mu layers to x to get predicted mus
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1) ##note that these are LeakyReLU and should continue as such, otherwise you have to remove them from the last layer
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # Apply sig layers to x to get predicted sigs - same as above
        if self.predict_sig:
            logsig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1) ##note that these are LeakyReLU and should continue as such, otherwise you have to remove them from the last layer
                logsig.append(x1)
            logsig=cat(logsig, dim=1) # DONT TAKE ABS, INSTEAD ASSUME PREDICTION IS OF LOG SIGMA, TAKE EXP OF MODEL PREDICTION

        # if self.predict_cov:
        #     rho=[]
        #     for norm, decode in zip(self.rho_norms, self.rhos):
        #         x1=clone(x)
        #         for n, d in zip(norm, decode):
        #             x1=d(n(x1))
        #             x1=self.decode_act(x1) ##note that these are LeakyReLU and should continue as such, otherwise you have to remove them from the last layer
        #         rho.append(x1)
        #     rho=cat(rho, dim=1)
        #     cov = # need to construct from rho and sig, right?
        #     cov = clamp(tanh(cov), min=-0.999, max=0.999)
        
        if self.predict_sig:
            return x_out, logsig
        if self.predict_cov:
            return x_out, cov
        else:
            return x_out
    

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
# Simple model for predicting target distributions from ONLY final halos
# EITHER make sure sigmas end with act func that makes them allways positive OR (better) interpret as log sigmas
# Why do sigmas go to nan?
#   Outputs of first linear get very large, which means sigmoid of them is always ~1 (would be the same for tanh)
#   But I really shouldnt be scaling to [0, 1] at all unless I want to scale targets too and rescale them both after
############################
        
class SimpleDistNet(nn.Module):
   '''
   Note - does include LayerNorm
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

    return mu, logsig
   
class SimpleDistNet2(nn.Module):
    '''
    Expansion of Chrisitan's MLP model to include a second output for the standard deviation of the target
    Also, allow for hyperparameters to be passed in. Remove LayerNorm option for now. 
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