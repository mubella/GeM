import sys

import torch

from layers import *

import torch.nn.functional as F
from Lstm_VAE import *
from DGCN import *
from graphMAE import *
from t2v import Time2Vec


class Model(nn.Module):
    # def __init__(self, t_dim, l_dim, u_dim, ex, hp, dropout=0.1):
    def __init__(self, t_dim, l_dim, u_dim, ex, hp):
        super(Model, self).__init__()

        embed_dim, w_ag, w_poi, nhid, ninput, noutput, dropout_GCN, Node_nhid = hp

        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl
        self.Time2Vec = Time2Vec('sin', out_dim=embed_dim)

        emb_l2 = nn.Embedding(l_dim, ninput, padding_idx=0)
        self.POI_feature_emb = POI_feature_emb(emb_l2, embed_dim)   #(p1,p2...pL)
        self.NodeAttnMap = NodeAttnMap(in_features=noutput, nhid=Node_nhid, use_mask=False)  #图注意力  ALB 2: 2

        self.POI_trans = POI_trans(l_max=l_dim - 1, wa=w_poi)             #ALB 1:w_poi=0
        self.MF = QKW(emb_l, emb_u, l_dim)


        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers, self.Time2Vec)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)

        embed_dim_lstm = embed_dim

        self.Lstm_u = LSTM_u(embed_dim, embed_dim_lstm, num_layers=1)

        self.GS_loc = GS_loc(emb_l, l_dim, embed_dim, l_dim)
        b = 0.5
        self.GS = GS(b)
        self.ac_fc = nn.ELU(inplace=False)
        b_dst = 50
        self.GS_Dst2 = GS_Dst2(embed_dim, b_dst)
        w_ag = 0.6
        self.Ag_layer2 = Ag_layer2(w=w_ag)
    def forward(self, traj, mat1, mat2, vec, traj_len, A, raw_X, ):  # mat2s_L,mat2s_S

        '''module 1:mulit-embed'''
        joint, delta, time_embeddings, loc_embeddings, user_embeddings, time2v = self.MultiEmbed(traj, mat1,
                                                                                                 traj_len)  # (N, M, emb), (N, M, M, emb)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)

        user_u = self.Lstm_u(self_attn, traj_len)
        user_u = user_u.to(device)
        loc_u, loc_v, loc_candites = self.GS_loc(traj, traj_len)  # poi的高斯分布

        user_u = user_u.squeeze(0)

        '''module 2:MA-dst distribution'''
        L_dst = self.GS_Dst2(user_u, loc_u, loc_v)
        prob_Gs = L_dst
        '''module 3:transition map'''
        X = self.POI_feature_emb(raw_X)  # X (L,4+embed_dim) 54
        X = X.to(device)
        A = A.to(device)
        # 使用Node转移矩阵修正最终概率
        attMap_e = self.NodeAttnMap(X, A, mat2)
        # attMap_e=self.NodeAttnMap2(X, A)
        PG_AV_W_output = self.POI_trans(prob_Gs, attMap_e, traj, traj_len)
        user = traj[:, 0, 0]
        UVW = self.MF(user, PG_AV_W_output)

        '''Final Ag'''
        # 最终poi图概率和用户attention概率聚合
        Add_prob = self.Ag_layer2(prob_Gs, UVW)
        GeM_prob = Add_prob.squeeze(0)

        return GeM_prob


class dst(nn.Module):
    def __init__(self):
        super(dst, self).__init__()

    def forward(self, user_u, user_v, loc_u, loc_v):
        dst = get_KL(user_u, user_v, loc_u, loc_v)
        dst = dst.squeeze()
        re_dst = max_min(dst)
        return dst


def get_WS(user_u, user_v, loc_u, loc_v):
    p1 = torch.sum(torch.pow((user_u - loc_u), 2), -1)
    p2 = torch.sum(torch.pow(torch.pow(user_v, 1 / 2) - torch.pow(loc_v, 1 / 2), 2), -1)
    dst = p1 + p2
    return dst


def get_KL(user_u, user_v, loc_u, loc_v):
    a = (torch.log(torch.prod(loc_v / user_v, -1)))
    b = torch.prod((1.0 / loc_v * user_v), -1)
    c = torch.sum(torch.pow((loc_u - user_u), 2) * (1.0 / loc_v), -1)
    n = 1
    dst = 0.5 * (a - n + b + c)
    return dst


class GS_loc(nn.Module):
    def __init__(self, emb_l, l_dim, embed_dim, l_max):
        super(GS_loc, self).__init__()
        self.emb_loc_u = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        self.emb_loc_v = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        self.ac_fc = nn.ELU(inplace=False)
        self.loc_max = l_max - 1

    def forward(self, traj, traj_len):
        candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long().to(device)
        # 压缩到[L]
        emb_loc_u = self.emb_loc_u(candidates)
        emb_loc_v = self.emb_loc_v(candidates)
        emb_loc_v = self.ac_fc(emb_loc_v) + 1
        return emb_loc_u, emb_loc_v, candidates  # loc_unq


def get_ma(x, mu, var):
    A = (x - mu)
    var_hat = 1.0 / var
    res = torch.sum(A * var_hat * A, dim=-1)  # 这里自动广播，不用区分两种情况
    ma_dist = torch.sqrt(res)  # 高斯分布中的协方差矩阵等于对角矩阵，可以推导出，等价于对距离进行加权(都是点乘)

    return ma_dist


def get_ma2(x, mu, var):
    A = (x - mu)
    var_hat = 1.0 / var
    # var_hat = var
    res = torch.sum(A * var_hat * A, dim=-1)  # 这里自动广播，不用区分两种情况
    #  print(res)
    right = torch.exp(-res / 2)
    left = 1 / torch.prod(var)
    ma_dist = left * right  # 高斯分布中的协方差矩阵等于对角矩阵，可以推导出，等价于对距离进行加权(都是点乘)

    return ma_dist


def get_o(x, mu):
    A = (x - mu)
    res = torch.sum(A * A, dim=-1)  # 这里自动广播，不用区分两种情况
    o_dist = torch.sqrt(res)  # 高斯分布中的协方差矩阵等于对角矩阵，可以推导出，等价于对距离进行加权(都是点乘)
    return o_dist


def max_min(x):
    min_x = torch.min(x)
    max_x = torch.max(x)
    re_x = (x - min_x) / (max_x - min_x)  # 越远概率越高

    return re_x


def max_min2(x):
    min_x = torch.min(x)
    max_x = torch.max(x)
    re_x = (max_x - x) / (max_x - min_x)  # 越远概率越高

    return re_x


def dst_count(x):
    count_list = [0, 0, 0, 0, 0]
    for item in x:
        if item < 0.1:
            count_list[0] = count_list[0] + 1
        if 0.1 < item < 0.3:
            count_list[1] = count_list[1] + 1
        if 0.3 < item < 0.6:
            count_list[2] = count_list[2] + 1
        if 0.6 < item < 0.9:
            count_list[3] = count_list[3] + 1
        if 0.9 < item < 1:
            count_list[4] = count_list[4] + 1


class GS_Dst2(nn.Module):
    def __init__(self, embed_dim, b):
        super(GS_Dst2, self).__init__()
        self.emb_dim = embed_dim
        self.b_dst = b

    def forward(self, user_u, loc_u, loc_v):  # [D],[L,D],[L]
        ma_dst = get_ma(user_u, loc_u, loc_v)
        re_dst = max_min(ma_dst)

        return re_dst  # ma_dst

def get_dot(x, mu):
    dot_dst = torch.sum( x * mu, dim=-1)  # 这里自动广播，不用区分两种情况
    return dot_dst


class GS_Dst(nn.Module):
    def __init__(self, embed_dim, b):
        super(GS_Dst, self).__init__()
        self.emb_dim = embed_dim
        self.beta = b

    def forward(self, user_u, user_v, loc_u, loc_v, output):  # [D],[L,D],[L]
        ma_dst = get_ma(user_u, loc_u, loc_v)
        return 1 / ma_dst  # ma_dst


class GS(nn.Module):
    def __init__(self, b):
        super(GS, self).__init__()
        self.beta = b

    def forward(self, output, L_dst):
        L_dst2 = L_dst.unsqueeze(0)
        min_x = torch.min(output)
        max_x = torch.max(output)
        Gs_prob = self.beta * output + L_dst2

        return Gs_prob  # L_dst [1,L]


def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, float):
        return (x, x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def reparameterization(mu, log_var):
    """
    Given a standard gaussian distribution epsilon ~ N(0,1),
    we can sample the random variable z as per z = mu + sigma * epsilon
    :param mu:
    :param log_var:
    :return: sampled z
    """
    sigma = torch.exp(log_var * 0.5)
    eps = torch.randn_like(sigma)  # sample
    return mu + sigma * eps  # 点乘