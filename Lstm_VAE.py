import sys

import numpy as np

from load import *
import torch
from torch import nn
from torch.nn import functional as F

seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda'
import  os


class LSTM_u(nn.Module):
    def __init__(self,emb_dim,emb_dim_Gs,num_layers):
        super(LSTM_u, self).__init__()
        self.emb_dim=emb_dim
        self.emb_dim_Gs = emb_dim_Gs
        self.Lstm=nn.LSTM(emb_dim,emb_dim_Gs,num_layers)

    def forward(self, self_attn,traj_len):
        self_attn=self_attn.permute(1,0,2)  #(M,N,emb)     对于M：这里应该截取前traj_len个 0:traj_len

        N = self_attn.shape[1]
        emb = self_attn.shape[2]
        NA = torch.zeros(N, 1, emb)
        h0 = torch.zeros(1, 1, self.emb_dim_Gs)
        c0 = torch.zeros(1, 1, self.emb_dim_Gs)

        h0 = h0.to(device)
        c0 = c0.to(device)
        NA = NA.to(device)
        for i in range(N):
            self_attn_traj=self_attn[:traj_len[i],i,:]   #(traj_len,N,emb)  修改
            output, (hn, cn)=self.Lstm(self_attn_traj.unsqueeze(1),(h0,c0))
            hn = hn.permute(1,0,2)  # hn:(1,N,emb)
            NA[i] = hn

        return NA  # hn:(N,1,emb)

class LSTM_v(nn.Module):
    def __init__(self,emb_dim,emb_dim_Gs,num_layers):
        super(LSTM_v, self).__init__()
        self.emb_dim=emb_dim
        self.emb_dim_Gs = emb_dim_Gs
        self.Lstm=nn.LSTM(emb_dim,emb_dim_Gs,num_layers)
        self.ac_fc = nn.ELU(inplace=False)

    def forward(self, self_attn,traj_len):
        self_attn=self_attn.permute(1,0,2)  #(M,N,emb)

        N = self_attn.shape[1]
        emb = self_attn.shape[2]
        NA = torch.zeros(N, 1, emb)
        h0 = torch.zeros(1, 1, self.emb_dim_Gs)
        c0 = torch.zeros(1, 1, self.emb_dim_Gs)
        h0 = h0.to(device)
        c0 = c0.to(device)
        for i in range(N):
            self_attn_traj=self_attn[:traj_len[i],i,:]   #(traj_len,N,emb)
            output, (hn, cn)=self.Lstm(self_attn_traj.unsqueeze(1),(h0,c0))
            hn = hn.permute(1, 0, 2)  # hn:(1,N,emb)
            NA[i]=hn

        user_v = self.ac_fc(NA) + 1
        return user_v
        return NA  # hn:(N,1,emb)
class VAE(nn.Module):

    def __init__(self, input_dim=100, h_dim=100, z_dim=50):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):  #lstm_emb (N,emb*2)
        batch_size = x.shape[0]

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        x_hat = x_hat.view(batch_size, self.input_dim )

        return x_hat.to(device), mu.to(device), log_var.to(device)

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))  #去除第一个全连接层
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 点乘

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.relu(self.fc5(h))
        return x_hat


class RT_att(nn.Module):
    def __init__(self,k,max_len):
        super(RT_att, self).__init__()
        self.k=k
        self.value = nn.Linear(k, 1, bias=False)

    def forward(self,mu,log_var,emb_candidates):
        # 随机从隐变量的分布中取隐变量
        sigma = torch.exp(log_var * 0.5)
        for i in range(self.k):
          eps = torch.randn_like(sigma)
          z=mu + sigma * eps  # 点乘  z(N,emb)

          if(i==0):
              k_z=z
          else:
              k_z=torch.cat([k_z,z],dim=1)
        attn_VAE = torch.bmm(emb_candidates, k_z.transpose(-1, -2))  # (N, L, k)

        attn_VAE=F.relu(torch.squeeze(self.value(attn_VAE)) )     #(N,L)

        return attn_VAE.to(device)

    def target_distribution(self, preds):
        targets = preds**2 / preds.sum(dim=0)
        targets = (targets.t() / targets.sum(dim=1)).t()
        return targets


class Ag_layer(nn.Module):
    def __init__(self,w):
        super(Ag_layer, self).__init__()
        self.w=w

    def forward(self,att_out,output):
        Final_output=torch.add(att_out,self.w*output)

        return  Final_output.to(device)

class Ag_layer2(nn.Module):
    def __init__(self,w):
        super(Ag_layer2, self).__init__()
        self.w=w

    def forward(self,att_out,output):
        Final_output=torch.add(att_out,self.w*output)
        return Final_output.to(device)
