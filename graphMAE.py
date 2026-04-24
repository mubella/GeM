from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)


    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class PreModel(nn.Module):
    def __init__(
            self,
            gcn_decoder,
            in_dim: int,
            mask_rate: float = 0.3,
            loss_fn: str = "mse",#mse sce
            replace_rate: float = 0.1,
            alpha_l: float = 1.1,

    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self.decoder=gcn_decoder

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]#L
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token


        return  out_x, (mask_nodes, keep_nodes)

    def forward(self, A, x,encoder):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(A, x,encoder)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, A, x,encoder):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise( x, self._mask_rate)
        enc_rep = encoder(use_x,A)
        # ---- attribute reconstruction ----

        rep = enc_rep
        rep[mask_nodes] = 0

        recon = self.decoder(rep,A)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss


