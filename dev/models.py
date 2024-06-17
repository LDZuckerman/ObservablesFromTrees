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

class MLP(Module):
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super().__init__()
        '''Simple MLP class with ReLU activiation + layernorm'''
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

class Sage(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, get_sig=False, get_cov=False, agg='sum'): # get_rho=False, 
        super(Sage, self).__init__()
        '''Model built upon the GraphSAGE convolutional layer. This is a node only model (no global, no edge).
        Model takes a data object from a dataloader in the forward call and takes out the rest itself. 
        hidden_channels, n_in, n_out must be specified
        Most other things can be customized at wish, e.g. activation functions for which ReLU and LeakyReLU can be used'''
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

        #################
        # Decode Layers #
        #################       

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

        #############################
        # Standard deviation Layers #
        #############################

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

        #get the data
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = x.float() # converts from float64 (e.g. "double") to float32 (which model wants)

        #encode
        if self.encode:
            x = self.node_enc(x)
        
        #convolutions 
        for conv in self.convs:
            x = conv(x, edge_index)
            x=self.conv_act(x) ##choose whichever
        if self.agg=='sum':  ## sum for physics
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        #decoder
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1) ##note that these are LeakyReLU and should continue as such, otherwise you have to remove them from the last layer
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # variance
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
    

######################################
### Make own MetaLayer-based Class ###
######################################

node_aggregation = scatter_sum  # scatter_mean
global_aggregation = scatter_sum  # scatter_mean

class EdgeModel(Module):
    def __init__(self, hidden):
        super(EdgeModel, self).__init__()
        self.mlp = MLP(hidden * 4, hidden, layer_norm=True)

    def forward(self, src, dest, edge_attr, u, batch): #forward should include everything
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs. ##what is B??
        # batch: [E] with max entry B - 1.
        cur_state = cat([src, dest, edge_attr, u[batch]], 1)
        return edge_attr + self.mlp(cur_state)

class EdgeModel2(Module):
    def __init__(self, hidden):
        super(EdgeModel2, self).__init__()
        self.mlp = MLP(3*hidden, hidden, layer_norm=True)

    def forward(self, src, dest, edge_attr, u, batch): #forward should include everything
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs. ##what is B??
        # batch: [E] with max entry B - 1.
        cur_state = cat([src, dest, edge_attr], 1)
        return edge_attr + self.mlp(cur_state)


class NodeModel(Module):
    def __init__(self, hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden * 2, hidden, layer_norm=True)
        self.node_mlp_2 = MLP(hidden * 3, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch): #forward should include everything
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))
        out = cat([x, out, u[batch]], dim=1)
        return x + self.node_mlp_2(out)

class NodeModel2(Module):
    def __init__(self, hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden , hidden, layer_norm=True)
        self.node_mlp_2 = MLP(hidden * 2, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch): #forward should include everything
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))
        out = cat([x, out], dim=1)
        return x + self.node_mlp_2(out)

class NodeNodeModel(Module):
    def __init__(self, hidden):
        super(NodeNodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden, hidden, layer_norm=True)
        self.node_mlp_2 = MLP(hidden, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch): #forward should include everything
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = x[row]
        out = self.node_mlp_1(out)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))
        out = x
        return x + self.node_mlp_2(out)

###############################################################################################
## If one is interested in doing some decoding on the global model / put in global features ###
###############################################################################################

class GlobalModel(Module): 
    def __init__(self, hidden):
        super(GlobalModel, self).__init__()
        self.global_mlp = MLP(hidden * 2, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch): #forward should include everything
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = cat([u, global_aggregation(x, batch, dim=0)], dim=1)
        return u + self.global_mlp(out) # do these just add on top of each other

class GlobalModelMulti(Module):
    def __init__(self, hidden):
        super(GlobalModelMulti, self).__init__()
        self.global_mlp = MLP(hidden * 5, hidden, layer_norm=True)
        '''Global model with many different global feats'''

    def forward(self, x, edge_index, edge_attr, u, batch): #forward should include everything
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        #takes a series of different global things into consideration
        s=scatter_sum(x, batch, dim=0)
        me=scatter_mean(x, batch, dim=0)
        mi=scatter_min(x, batch, dim=0)[0]
        ma=scatter_max(x, batch, dim=0)[0]
        std=scatter_mean(square(x), batch, dim=0)-square(me)
        concat = cat([u, s, mi, ma, std], dim=1)
        return u + self.global_mlp(concat) ## still a bit in doubt over if this should be a sum

###############################################################################################

# Put it all together like https://arxiv.org/pdf/1806.01261.pdf

class MetaMulti(Module):
    def __init__(self, hidden_states, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=1, decode_activation='none', layernorm=True):
        super(self.__class__, self).__init__()
        '''This model is in early attempt, it's a bit slow but has higher levels of interpretability and should theoretically be better'''
        hidden=hidden_states
        n_in=in_channels
        n_out=out_channels
        self.node_enc = MLP(n_in, hidden, layer_norm=True)
        self.edge_enc = MLP(3, hidden, layer_norm=True)
        self.decoder = MLP(hidden, n_out)
        self.ops = ModuleList(
            [
                MetaLayer(edge_model=EdgeModel(hidden), node_model=NodeModel(hidden), global_model=GlobalModelMulti(hidden))
                for _ in range(conv_layers)
            ]
        )
        self.hidden = hidden
        self.norm_out=LayerNorm(normalized_shape=self.hidden)

    
    def forward(self, graph):
        x = self.node_enc(graph.x)  # Take all feats and encode
        e_feat = graph.x[:,[0,3]] # scale factor and virial mass
        adj = graph.edge_index
        e_encode=cat([graph.edge_attr.view(-1,1), e_feat[adj[0]] - e_feat[adj[1]]], -1)
        
        e = self.edge_enc(e_encode) #put in edge_attr
        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden, device=x.device, dtype=float32
        )
        batch = graph.batch
        for op in self.ops:
            x, e, u = op(x, adj, e, u, batch)
        x = global_add_pool(x, batch)
        
        out = self.decoder(self.norm_out(x))

        return out

class Meta(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, variance=0, agg='sum', rho=0):
        super(Meta, self).__init__()
        '''Same as above but without the multi concat '''
        self.encode=encode
        self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True)
        self.edge_enc = MLP(3, hidden_channels, layer_norm=True)
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.variance=variance
        self.agg=agg
        self.rho=rho
        ########################
        # Convolutional Layers #
        ######################## 

        self.convs=ModuleList()
        self.convs.append(MetaLayer(edge_model=EdgeModel(hidden_channels), node_model=NodeModel(hidden_channels), global_model=GlobalModelMulti(hidden_channels)))
        for _ in range(int(conv_layers-1)):
            self.convs.append(MetaLayer(edge_model=EdgeModel(hidden_channels), node_model=NodeModel(hidden_channels), global_model=GlobalModelMulti(hidden_channels)))

        ##################
        # Decode Layers #
        ##################       

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

        ###################
        # Variance Layers #
        ###################

        if variance:
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

        if self.rho!=0:
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
            print('RelU conv activation')
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            print('LeakyRelU conv activation')
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            print('RelU decode activation')
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            print('LeakyRelU decode activation')
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        #get the data
        x = self.node_enc(graph.x)  # Take all feats and encode
        e_feat = graph.x[:,[0,3]] # scale factor and virial mass
        adj = graph.edge_index
        e_encode=cat([graph.edge_attr.view(-1,1), e_feat[adj[0]] - e_feat[adj[1]]], -1)
        
        e = self.edge_enc(e_encode) #put in edge_attr
        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden_channels, device=x.device, dtype=float32
        )

        #convolutions 
        
        batch = graph.batch
        for op in self.convs:
            x, e, u = op(x, adj, e, u, batch)
        if self.agg=='sum':
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        #decoder
        
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1)
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # variance
        if self.variance:
            sig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                sig.append(x1)
            sig=abs(cat(sig, dim=1))

        if self.rho!=0:
            rho=[]
            for norm, decode in zip(self.rho_norms, self.rhos):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                rho.append(x1)
            rho=abs(cat(rho, dim=1)) ### not sure this works with only 1d
        
        if self.variance:
            if self.rho!=0:
                return x_out, sig, tanh(rho)
            else:
                return x_out, sig
        else:
            return x_out

class MetaEdge2(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, variance=0, agg='sum', rho=0):
        super(MetaEdge2, self).__init__()
        ''' Same as above but no global model'''
        self.encode=encode
        self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True)
        self.edge_enc = MLP(hidden_channels, hidden_channels, layer_norm=True)
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.variance=variance
        self.agg=agg
        self.rho=rho
        ########################
        # Convolutional Layers #
        ######################## 

        self.convs=ModuleList()
        self.convs.append(MetaLayer(edge_model=EdgeModel2(hidden_channels), node_model=NodeModel(hidden_channels), global_model=None))
        for _ in range(int(conv_layers-1)):
            self.convs.append(MetaLayer(edge_model=EdgeModel2(hidden_channels), node_model=NodeModel(hidden_channels), global_model=None))

        ##################
        # Decode Layers #
        ##################       

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

        ###################
        # Variance Layers #
        ###################

        if variance:
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

        if self.rho!=0:
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
            print('RelU conv activation')
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            print('LeakyRelU conv activation')
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            print('RelU decode activation')
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            print('LeakyRelU decode activation')
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        #get the data
        x = self.node_enc(graph.x)  # Take all feats and encode
        # e_feat = graph.x[:,[0,3]] # scale factor and virial mass
        adj = graph.edge_index
        e_encode=scatter_add(x[adj[0]],adj[1], dim=0)

        # print(e_encode)
        # print(e_encode.shape)

        e = self.edge_enc(e_encode) #put in edge_attr

        # print(e)
        # print(e.shape)

        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden_channels, device=x.device, dtype=float32
        )

        #convolutions 
        
        batch = graph.batch
        for op in self.convs:
            x, e, _ = op(x, adj, e, u, batch)
        if self.agg=='sum':
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        #decoder
        
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1)
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # variance
        if self.variance:
            sig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                sig.append(x1)
            sig=abs(cat(sig, dim=1))

        if self.rho!=0:
            rho=[]
            for norm, decode in zip(self.rho_norms, self.rhos):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                rho.append(x1)
            rho=abs(cat(rho, dim=1)) ### not sure this works with only 1d
        
        if self.variance:
            if self.rho!=0:
                return x_out, sig, tanh(rho)
            else:
                return x_out, sig
        else:
            return x_out

class MetaNode(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, variance=0, agg='sum', rho=0):
        super(MetaNode, self).__init__()
        '''Same as above but with only node'''
        self.encode=encode
        self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True)
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.variance=variance
        self.agg=agg
        self.rho=rho
        ########################
        # Convolutional Layers #
        ######################## 

        self.convs=ModuleList()
        self.convs.append(MetaLayer(edge_model=None, node_model=NodeNodeModel(hidden_channels), global_model=None))
        for _ in range(int(conv_layers-1)):
            self.convs.append(MetaLayer(edge_model=None, node_model=NodeNodeModel(hidden_channels), global_model=None))

        ##################
        # Decode Layers #
        ##################       

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

        ###################
        # Variance Layers #
        ###################

        if variance:
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

        if self.rho!=0:
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
            print('RelU conv activation')
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            print('LeakyRelU conv activation')
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            print('RelU decode activation')
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            print('LeakyRelU decode activation')
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        #get the data
        x = self.node_enc(graph.x)  # Take all feats and encode
        adj = graph.edge_index

        e = graph.edge_attr.view(-1,1) # scale factor and virial mass
        # e_encode=cat([graph.edge_attr.view(-1,1), e_feat[adj[0]] - e_feat[adj[1]]], -1)
        # e = self.edge_enc(e_encode) #put in edge_attr

        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden_channels, device=x.device, dtype=float32
        )

        #convolutions 
        
        batch = graph.batch
        for op in self.convs:
            x, _, _ = op(x, adj, e, u, batch)
        if self.agg=='sum':
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        #decoder
        
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1)
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # variance
        if self.variance:
            sig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                sig.append(x1)
            sig=abs(cat(sig, dim=1))

        if self.rho!=0:
            rho=[]
            for norm, decode in zip(self.rho_norms, self.rhos):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                rho.append(x1)
            rho=abs(cat(rho, dim=1)) ### not sure this works with only 1d
        
        if self.variance:
            if self.rho!=0:
                return x_out, sig, tanh(rho)
            else:
                return x_out, sig
        else:
            return x_out

class PysrNet(Module):
    def __init__(self, n_outs=3, hidden_channels=64, n_feat=5, n_targ=1):
        super(PysrNet, self).__init__()
        self.g1 = MLP(n_feat, n_outs, hidden = hidden_channels)
        self.g2 = MLP(n_outs, n_outs,  hidden = hidden_channels)
        self.g3= MLP(n_outs, n_outs, hidden = hidden_channels) 
    
        self.f = MLP(n_outs, n_targ,  hidden = hidden_channels)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        
        x = self.g1(x) # NODE ENCODER
        
#         global adj, batch1, xe
        adj = edge_index
#         global neighbours
#         global N_sum
#         neighbours = x
#         batch1=batch
        
        N_sum = scatter_add(x[adj[0]],adj[1], dim=0) #ADD NEIGHBORHOOD NODES
        xe = self.g2(N_sum) #ENCODE EDGE SUM
        x[adj[1]]+=xe[adj[1]] #only add where we have receiving nodes TO UPDATA X
        x = self.g3(x) #MLP ON EDGE ADDED FEATURES

        x = global_add_pool(x, batch)

        x = self.f(x)

        return x


class MetaEdge(Module):
    def __init__(self, hidden_channels, in_channels, out_channels, encode=True, conv_layers=3, conv_activation='relu', 
                    decode_layers=2, decode_activation='none', layernorm=True, variance=0, agg='sum', rho=0):
        super(MetaEdge, self).__init__()
        ''' Same as above but no global model'''
        self.encode=encode
        self.node_enc = MLP(in_channels, hidden_channels, layer_norm=True)
        self.edge_enc = MLP(3, hidden_channels, layer_norm=True)
        self.decode_activation=decode_activation
        self.conv_activation=conv_activation
        self.layernorm=layernorm
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.variance=variance
        self.agg=agg
        self.rho=rho
        ########################
        # Convolutional Layers #
        ######################## 

        self.convs=ModuleList()
        self.convs.append(MetaLayer(edge_model=EdgeModel(hidden_channels), node_model=NodeModel(hidden_channels), global_model=None))
        for _ in range(int(conv_layers-1)):
            self.convs.append(MetaLayer(edge_model=EdgeModel(hidden_channels), node_model=NodeModel(hidden_channels), global_model=None))

        ##################
        # Decode Layers #
        ##################       

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

        ###################
        # Variance Layers #
        ###################

        if variance:
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

        if self.rho!=0:
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
            print('RelU conv activation')
            act = ReLU()
            return act
        if self.conv_activation =='leakyrelu':
            print('LeakyRelU conv activation')
            act=LeakyReLU()
            return act
        if not self.conv_activation:
            raise ValueError("Please specify a conv activation function")

    def decode_act_f(self):
        if self.decode_activation =='relu':
            print('RelU decode activation')
            act = ReLU()
            return act
        if self.decode_activation =='leakyrelu':
            print('LeakyRelU decode activation')
            act=LeakyReLU()
            return act
        if not self.decode_activation:
            print("Please specify a decode activation function")
            return None

    def forward(self, graph):

        #get the data
        x = self.node_enc(graph.x)  # Take all feats and encode
        e_feat = graph.x[:,[0,3]] # scale factor and virial mass
        adj = graph.edge_index
        e_encode=cat([graph.edge_attr.view(-1,1), e_feat[adj[0]] - e_feat[adj[1]]], -1)
        
        e = self.edge_enc(e_encode) #put in edge_attr
        # Initialize global features as 0:
        u = zeros(
            graph.batch[-1] + 1, self.hidden_channels, device=x.device, dtype=float32
        )

        #convolutions 
        
        batch = graph.batch
        for op in self.convs:
            x, e, _ = op(x, adj, e, u, batch)
        if self.agg=='sum':
            x = global_add_pool(x, batch)
        if self.agg=='max':
            x = global_max_pool(x, batch)
        
        #decoder
        
        x_out=[]
        for norm, decode in zip(self.norms, self.decoders):
            x1=clone(x)
            for n, d in zip(norm, decode):
                x1=d(n(x1))
                x1=self.decode_act(x1)
            x_out.append(x1)
        x_out=cat(x_out, dim=1)
        
        # variance
        if self.variance:
            sig=[]
            for norm, decode in zip(self.sig_norms, self.sigs):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                sig.append(x1)
            sig=cat(sig, dim=1) # DONT TAKE ABS, INSTEAD ASSUM PREDICTION IS OF LOG SIGMA, TAKE EXP OF MODEL PREDICTION
            
        if self.rho!=0:
            rho=[]
            for norm, decode in zip(self.rho_norms, self.rhos):
                x1=clone(x)
                for n, d in zip(norm, decode):
                    x1=d(n(x1))
                    x1=self.decode_act(x1)
                rho.append(x1)
            rho=cat(rho, dim=1)
        
        if self.variance:
            if self.rho!=0:
                return x_out, sig, tanh(rho)
            else:
                return x_out, sig
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
# Simple model for predicting target distributions from ONLY final halo properties
# EITHER make sure sigmas end with act func that makes them allways positive OR (better) interpret as log sigmas
# Why do sigmas go to nan?
#   Outputs of first linear get very large, which means sigmoid of them is always ~1 (would be the same for tanh)
#   But I really shouldnt be scaling to [0, 1] at all unless I want to scale targets too and rescale them both after
############################
        
class SimpleDistNet(nn.Module):
    
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
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super().__init__()
        '''Simple MLP class with ReLU activation + layernorm'''
        layers = [nn.Linear(n_in, hidden), nn.ReLU()]
        for i in range(nlayers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.Linear(hidden, 2 * n_out))  # Output both mean and std
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)
        n_out = output.size(1) // 2
        mean = output[:, :n_out]
        std = F.softplus(output[:, n_out:])  # Ensure std is positive
        return mean, std
   