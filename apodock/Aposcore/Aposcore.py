import torch
import torch.nn as nn
from torch.nn import Linear
from apodock.Aposcore.HIL import GVP_embedding, InteractionBlock, MPNNL
from torch_geometric.utils import to_dense_batch


class Aposcore(nn.Module):
    def __init__(
        self,
        node_dim,
        hidden_dim,
        num_heads,
        dropout,
        crossAttention,
        atten_active_fuc,
        num_layers,
        interact_type,
        n_gaussians=10,
    ):
        super().__init__()
        self.interact_type = interact_type

        self.lin_node_lig = nn.Sequential(Linear(node_dim, hidden_dim), nn.LeakyReLU())

        self.conv = nn.ModuleList(
            [MPNNL(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )

        self.conv_protein = GVP_embedding(
            (6, 3),
            (hidden_dim, 16),
            (32, 1),
            (32, 1),
            seq_in=True,
            num_layers=num_layers,
            drop_rate=dropout,
            plm=False,
        )
        self.interaction = InteractionBlock(
            hidden_dim,
            num_heads,
            dropout,
            atten_active_fuc,
            crossAttention,
            interact_type,
        )
        self.z_pi = nn.Sequential(
            nn.Linear(hidden_dim, n_gaussians),
            nn.Softmax(dim=-1),
        )  # pi:
        self.z_mu = nn.Sequential(
            nn.Linear(hidden_dim, n_gaussians),
            nn.LeakyReLU(),
        )
        self.z_sigma = nn.Sequential(
            nn.Linear(hidden_dim, n_gaussians),
            nn.LeakyReLU(),
        )

    def forward(self, data):
        data_l = data["ligand_features"]
        data_aa = data["protein_features"]

        (
            x_l,
            edge_index_l,
            x_aa,
            seq,
            node_s,
            node_v,
            edge_index,
            edge_s,
            edges_v,
            edge_feat_l,
            pos_l,
            pos_p,
        ) = (
            data_l.x,
            data_l.edge_index,
            data_aa.x_aa,
            data_aa.seq,
            data_aa.node_s,
            data_aa.node_v,
            data_aa.edge_index,
            data_aa.edge_s,
            data_aa.edge_v,
            data_l.edge_attr,
            data_l.pos,
            data_aa.pos,
        )
        # print('x_l:', x_l.shape)
        x_l = self.lin_node_lig(x_l)
        for conv in self.conv:
            x_l = conv(x_l, edge_index_l, edge_feat_l)

        nodes = (node_s, node_v)
        edges = (edge_s, edges_v)
        protein_out = self.conv_protein(nodes, edge_index, edges, seq)
        x_l, mask_l = to_dense_batch(x_l, data_l.batch, fill_value=0)
        protein_out, mask_p = to_dense_batch(protein_out, data_aa.batch, fill_value=0)
        pos_l, _ = to_dense_batch(pos_l, data_l.batch, fill_value=0)
        pos_p, _ = to_dense_batch(pos_p, data_aa.batch, fill_value=0)
        (B, N_l, _), N_p = x_l.size(), protein_out.size(1)
        Interact, Interact_mask = self.interaction(x_l, protein_out, mask_l, mask_p)
        self.B = B
        self.N_l = N_l
        self.N_p = N_p

        ## Get batch indexes for ligand-target combined features
        device = x_l.device  # Get device from input tensor
        C_batch = (
            torch.tensor(range(B), device=device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, N_l, N_p)
        )
        C_batch = C_batch[Interact_mask]

        dist = self.compute_euclidean_distances_matrix(pos_l, pos_p.view(B, -1, 3))[
            Interact_mask
        ]
        # print(self.compute_euclidean_distances_matrix(pos_l, pos_p.view(B,-1,3)).shape)
        pi = self.z_pi(Interact)
        sigma = self.z_sigma(Interact) + 1.1
        sigma = torch.clamp(sigma, min=1e-6)
        mu = self.z_mu(Interact) + 1

        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch

    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.double()
        Y = Y.double()

        dists = (
            -2 * torch.bmm(X, Y.permute(0, 2, 1))
            + torch.sum(Y**2, axis=-1).unsqueeze(1)
            + torch.sum(X**2, axis=-1).unsqueeze(-1)
        )

        return torch.nan_to_num(
            (dists**0.5).view(self.B, self.N_l, -1, 24), 10000
        ).min(axis=-1)[0]

    def mdn_loss(self, pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):

        normal = torch.distributions.Normal(mu, sigma)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        prob = (torch.log(pi + eps1) + loglik).exp().sum(1)
        loss = -torch.log(prob + eps2)

        return loss, prob

    def calculate_probablity(self, pi, sigma, mu, y):
        normal = torch.distributions.Normal(mu, sigma)
        logprob = normal.log_prob(y.expand_as(normal.loc))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)
        return prob

    def sample(self, pi, sigma, mu):
        k = torch.multinomial(pi, 1).squeeze(-1)  # get the index of the gaussian
        indices = (
            torch.tensor(range(self.B)).unsqueeze(-1).repeat(1, self.N_l).unsqueeze(-1)
        )
        indices = indices[k == 1].squeeze(-1)
        mu = mu[indices]
        sigma = sigma[indices]
        normal = torch.distributions.Normal(mu, sigma)
        return normal.sample()  #


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
