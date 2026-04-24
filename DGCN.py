'''
X = self.POI_feature_emb(Raw_X)
X = self.GCN(X, A)
attMap = self.NodeAttMap(X, A)
# 下用X[index]代替emb_layer.emb_l
'''
import sys

import numpy as np

from load import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math

import torch.nn.functional as F

seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda'



class NodeAttnMap2(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap2, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        X=X.float()
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    
class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False): #in_features(L,emb)
        super(NodeAttnMap, self).__init__()
        self.emb_size=in_features
        self.query = nn.Linear(self.emb_size, nhid, bias=False)
        self.key = nn.Linear(self.emb_size, nhid, bias=False)
       # self.value = nn.Linear(self.emb_size, nhid, bias=False)
        self.use_mask = use_mask
        self.out_features = nhid
        # self.leakyrelu_l = nn.LeakyReLU(0.2)
        # self.leakyrelu_s = nn.LeakyReLU(0.2)

        self.leakyrelu = nn.LeakyReLU(0.2)
    def forward(self, X, A,delta):
        #a=torch.mm(self.query(X), self.key(X).T)
        a = torch.mm(self.query(X), self.key(X).transpose(-1, -2))
        e_normal=self.leakyrelu(a)

      #  print('e_normal',e_normal)
        if self.use_mask:
            e_normal= torch.where(A > 0, e_normal, torch.zeros_like(e_normal))  # mask
        #A = A + 1  # shift from 0-1 to 1~e
        A=torch.exp(A)
        attMap_e=e_normal*A

        return attMap_e

def max_min(x):
    min_x = torch.min(x)
    max_x = torch.max(x)
    re_x = (x-min_x) / (max_x - min_x)   #越远概率越高
    return re_x


class POI_trans(nn.Module):
    def __init__(self, l_max,wa): #in_features(L,emb)
        super(POI_trans, self).__init__()
        self.fuse_weight =wa
        self.L=l_max
        self.adjust2 = torch.nn.Parameter(torch.FloatTensor(1,self.L), requires_grad=True)
        nn.init.xavier_uniform_(self.adjust2.data, gain=1.414)

    def forward(self,Final_output,attMap_e,traj,traj_len):
        print(Final_output.shape)
        y_pred_poi_adjusted = torch.zeros_like(Final_output)#N,L

        for i in range(Final_output.shape[0]):
             prob1=attMap_e[traj[i, traj_len[i] - 1, 1] - 1, :]     #1-hop

             prob1_m = max_min(prob1)
             y_pred_poi_adjusted[i, :] = prob1_m
            
             if traj_len[i]>=2:
                 prob=attMap_e[traj[i,traj_len[i]-2,1]-1, :]
                 prob=torch.unsqueeze(prob,dim=0)
                 prob2=torch.mm(prob,attMap_e)   #二阶转移
                 prob2_m=max_min(prob2)
                 w=max_min(self.adjust2)
                 y_pred_poi_adjusted[i, :] = y_pred_poi_adjusted[i, :] + self.fuse_weight * torch.mul(w,prob2_m)

        return y_pred_poi_adjusted



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W) # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x #(L,emb)

class GCN_V(nn.Module):
    def __init__(self, ninput, nhid, noutput, noutput2,dropout):
        super(GCN_V, self).__init__()
        self.noutput2=noutput2
        self.gcn = nn.ModuleList()
        self.gcn_mu=GraphConvolution(noutput,noutput2)
        self.gcn_log = GraphConvolution(noutput,noutput2)

        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        self.mu=self.gcn_mu(x,adj)
        self.logstd=self.gcn_log(x,adj)
        gaussian_noise = torch.randn(x.size(0),self.noutput2).to(device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mu
        return sampled_z,self.mu,self.logstd#(L,emb)


class GCN_Decoder(nn.Module):
    def __init__(self, sigmoid=True):
        super(GCN_Decoder, self).__init__()
        self.sigmoid = True

    def forward(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class POI_feature_emb(nn.Module):
    def __init__(self, emb_loc,embed_dim):
        super(POI_feature_emb, self).__init__()
        self.emb_loc2 = emb_loc
        self.embed_dim=embed_dim

    def forward(self, raw_X):   #X[L,5]  (check_cnt,poi_index,delta_t,lat,lon)
        candidates=raw_X[:, 1].long()
        loc_emb = self.emb_loc2(candidates)  # (L) --> (l, embed)

        return loc_emb

class QKW(nn.Module):
    def __init__(self, emb_l, emb_u,l_dim):
        super(QKW, self).__init__()
        self.emb_loc = emb_l
        self.emb_user = emb_u
        self.loc_max=l_dim-1
    def forward(self, user,W):
        candidates = torch.linspace(1, int(self.loc_max),steps=self.loc_max).long()  # (L)
        candidates = candidates.to(device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (L, emb)
        emb_user=self.emb_user(user)  #(1,emb)
        QK=emb_user@emb_candidates.T  #(1,L)
        QKW=QK+W
        QKW=max_min(QKW)

        return QKW


