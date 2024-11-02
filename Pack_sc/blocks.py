
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sc_utils import gather_edges, gather_nodes, PositionalEncodings
from data_utils import calc_bb_dihedrals

# heterogeneous interaction layer
class MPNNL(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 dropout,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MPNNL, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(6, out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.updateNN = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.messageNN = nn.Sequential(
            nn.Linear(in_channels*3, out_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

    def forward(self, x, edge_index, edge_features,
                size=None):

        out_node_intra = self.propagate(edge_index=edge_index, x=x,edge_features=edge_features,messageNN=self.messageNN, updateNN=self.updateNN , size=size)#radial=radial_cov,
        out_node = self.mlp_node_cov(x + out_node_intra)  

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, messageNN, edge_features,
                index: Tensor):
        edge_feat = self.edge_mlp(edge_features)
        return messageNN(torch.cat([x_j, x_i,edge_feat], dim=-1)) 

    def update(self, aggr_out: Tensor, x: Tensor, updateNN):
        return updateNN(torch.cat([aggr_out, x], dim=-1))

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, num_heads, atten_active_fuc, dropout):
        super(AttentionBlock, self).__init__()
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.atten_active_fuc = atten_active_fuc
        self.head_dim = hid_dim // num_heads

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.sqrt_dk = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]

        # Transform Q, K, V
        q = self.f_q(query).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_q, head_dim]
        k = self.f_k(key).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N_k, head_dim]

        v = self.f_v(value).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_k, head_dim]


        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_dk  # [B, num_heads, N_q, N_k]
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print(mask.shape)
            # print(scores.shape)
            scores = scores.masked_fill(mask == 0, float('-inf'))
       
        # Apply softmax and dropout
        if self.atten_active_fuc == "softmax":
            attention_weights = F.softmax(scores, dim=-1)
        elif self.atten_active_fuc == "sigmoid":
            attention_weights = F.sigmoid(scores)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to the values
        weighted = torch.matmul(attention_weights, v)  # [B, num_heads, N_q, head_dim]
        # print(weighted.shape)
        weighted = weighted.permute(0, 2, 1, 3).contiguous() # [B, N_q, num_heads, head_dim]

        # Concatenate heads and put through final linear layer
        weighted = weighted.view(B, -1, self.hid_dim)   # [B, N_q, hid_dim]
        output = self.fc_out(weighted)

        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, hid_dim, dropout, atten_active_fuc, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim=hid_dim, num_heads=num_heads, atten_active_fuc=atten_active_fuc, dropout=dropout)
        
        self.linear_res = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.linear_lig = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.norm_aa = nn.LayerNorm(hid_dim)
        self.norm_lig = nn.LayerNorm(hid_dim)

    def forward(self, ligand_features, aa_features, mask_l, mask_aa):
        # 生成掩码以匹配attention分数的维度
        mask_l_expanded = mask_l.unsqueeze(1)  # [B, 1, N_l]
        mask_aa_expanded = mask_aa.unsqueeze(1)  # [B, 1, N_aa]

        # 交叉注意力计算
        aa_att = self.att(aa_features, ligand_features, ligand_features, mask=mask_l_expanded)
        lig_att = self.att(ligand_features, aa_features, aa_features, mask=mask_aa_expanded)
        # 线性变换与残差连接
        aa_features = self.linear_res(aa_att) + aa_features
        # aa_features = aa_att + aa_features
        aa_features = self.norm_aa(aa_features)
        
        
        ligand_features = self.linear_lig(lig_att)+ ligand_features #
        # ligand_features = lig_att + ligand_features

        ligand_features = self.norm_lig(ligand_features)
    
     

        return ligand_features, aa_features

class InteractionBlock(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout, atten_active_fuc, crossAttention=True, interact_type="product"):
        super(InteractionBlock, self).__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.interact_type = interact_type
        self.crossAttention = crossAttention
        if self.crossAttention:
            self.prot_lig_attention = CrossAttentionBlock(hid_dim=hid_dim, dropout=dropout, atten_active_fuc = atten_active_fuc, num_heads=num_heads)
        
        if self.interact_type == "product":

            self.mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hid_dim*2, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
            )
        

        self.lig_trans_inteact = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
        )

        self.aa_trans_inteact = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(),
        )

    def forward(self, lig_features, aa_features, lig_mask=None, aa_mask=None):
            # Apply transformations to ligand and amino acid features
            if self.crossAttention:
                lig_features, aa_features = self.prot_lig_attention(lig_features, aa_features, lig_mask, aa_mask)
            

            # print(aa_features.shape)
            # Extend features for matrix multiplication
            lig_features = lig_features.unsqueeze(2)  # [batch_size, N_l, 1, hidden]
            # print('lig_features_exp',lig_features_exp.shape)
            aa_features = aa_features.unsqueeze(1)  # [batch_size, 1, N_aa, hidden]
            # print('aa_features_exp',aa_features_exp.shape)
            if self.interact_type == "product":
                lig_features = self.lig_trans_inteact(lig_features)
                aa_features = self.aa_trans_inteact(aa_features)
                interaction = torch.multiply(lig_features, aa_features) 

            else:
                N_l =lig_features.size(1)
                N_p =aa_features.size(2)

                lig_features = self.lig_trans_inteact(lig_features)
                lig_features = lig_features.repeat(1,1,N_p,1)
                # print('lig_features',lig_features.shape)
                aa_features = self.aa_trans_inteact(aa_features)
                aa_features = aa_features.repeat(1,N_l,1,1)

                interaction = torch.cat((lig_features, aa_features), dim=-1) # [batch_size, N_l, N_aa, 2*hidden]

            if lig_mask is not None and aa_mask is not None:
                lig_mask_exp = lig_mask.unsqueeze(2)  # [batch_size, N_l, 1]
                aa_mask_exp = aa_mask.unsqueeze(1)  # [batch_size, 1, N_aa]
                interaction_mask = lig_mask_exp & aa_mask_exp  # [batch_size, N_l, N_aa]
            interaction = interaction[interaction_mask]
            interaction = self.mlp(interaction)
            return interaction, interaction_mask



class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features=128,
        node_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom37_order=False,
        device=None,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.num_positional_embeddings = num_positional_embeddings
        self.num_chain_embeddings = num_chain_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.atom37_order = atom37_order
        self.device = device
        self.atom_context_num = atom_context_num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # deal with oxygen index
        # ------
        self.N_idx = 0
        self.CA_idx = 1
        self.C_idx = 2

        if atom37_order:
            self.O_idx = 4
        else:
            self.O_idx = 3
        # -------
        self.positional_embeddings = PositionalEncodings(num_positional_embeddings)

        # Features for the encoder
        enc_node_in = 21 + 128# + 1536 # alphabet for the sequence + dihedral angles + ESM embeddings
        enc_edge_in = (
            num_positional_embeddings + num_rbf * 14 * 14
        )  # positional + distance features

        self.enc_node_in = enc_node_in
        self.enc_edge_in = enc_edge_in

        self.enc_edge_embedding = nn.Linear(enc_edge_in, edge_features, bias=False)
        self.enc_norm_edges = nn.LayerNorm(edge_features)
        self.enc_node_embedding = nn.Linear(enc_node_in, node_features, bias=False)
        self.enc_norm_nodes = nn.LayerNorm(node_features)

        # Features for the decoder
        dec_node_in = 14 * atom_context_num * num_rbf
        dec_edge_in = num_rbf * 14 * 14 + 42

        self.dec_node_in = dec_node_in
        self.dec_edge_in = dec_edge_in

        self.W_XY_project_down1 = nn.Linear(num_rbf + 120, num_rbf, bias=True)
        self.dec_edge_embedding1 = nn.Linear(dec_edge_in, edge_features, bias=False)
        self.dec_norm_edges1 = nn.LayerNorm(edge_features)
        self.dec_node_embedding1 = nn.Linear(dec_node_in, node_features, bias=False)
        self.dec_norm_nodes1 = nn.LayerNorm(node_features)

        self.node_project_down = nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = nn.LayerNorm(node_features)

        self.type_linear = nn.Linear(147, 64)

        self.y_nodes = nn.Linear(147, node_features, bias=False)
        self.y_edges = nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = nn.LayerNorm(node_features)
        self.norm_y_nodes = nn.LayerNorm(node_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx


    def _rbf(
        self,
        D,
        D_mu_shape=[1, 1, 1, -1],
        lower_bound=0.0,
        upper_bound=20.0,
        num_bins=16,
    ):
        device = D.device
        D_min, D_max, D_count = lower_bound, upper_bound, num_bins
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view(D_mu_shape)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(
        self,
        A,
        B,
        E_idx,
        D_mu_shape=[1, 1, 1, -1],
        lower_bound=2.0,
        upper_bound=22.0,
        num_bins=16,
    ):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(
            D_A_B_neighbors,
            D_mu_shape=D_mu_shape,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_bins=num_bins,
        )
        return RBF_A_B

    def features_encode(self, features):
        """
        make protein graph and encode backbone
        """
        S = features["S"]
        X = features["x"]
        # print(X.shape)
        mask = features["mask"]
        # print(mask.shape)
        atom14_mask = features["atom14_mask"]
        # print(atom14_mask.shape)
        X_m = atom14_mask*mask[:, :, None]  # [B,L,14]*[B,L,1] -> [B,L,14]
        # print(X_m.shape)
        R_idx = features["R_idx"]
        chain_labels = features["chain_labels"]
        protein_mpnn_feat = features["protein_mpnn_feat"]
        # esm_feat = features["esm_feat"]
        
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        Ca = X[:, :, self.CA_idx, :]
        N = X[:, :, self.N_idx, :]
        C = X[:, :, self.C_idx, :]
        O = X[:, :, self.O_idx, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        _, E_idx = self._dist(Ca, mask)

        X_m_gathered = gather_nodes(X_m, E_idx) 

        backbone_coords_list = [N, Ca, C, O, Cb]
        backbone_coords_list = [N, Ca, C, O]
        # print(backbone_coords_list)
        sc_coords_list = [X[:, :, i, :] for i in range(4, X.shape[-2])]
        # print(len(sc_coords_list))
        # print(sc_coords_list)
        # sc_coords_list = list(torch.split(sc_coords_list, 3, dim=-1))
        # print(backbone_coords_list.shape)
        all_coords_list = backbone_coords_list + sc_coords_list
        all_coords_list = torch.stack(all_coords_list, dim=-2)
        # print(all_coords_list.shape)
        # print(len(all_coords_list))
        RBF_all = []
        for atom_1 in range(14):
            for atom_2 in range(14):
                rbf_features=self._get_rbf(
                all_coords_list[:, :, atom_1, :],
                all_coords_list[:, :, atom_2, :],
                E_idx,
                D_mu_shape=[1, 1, 1, -1],
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                num_bins=self.num_rbf,
                )

                rbf_features = (
                rbf_features
                * X_m[:, :, atom_1, None, None]
                * X_m_gathered[:, :, :, atom_2, None]
                )
                RBF_all.append(rbf_features)
                

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)  # [B, L,30, n_atoms * n_atoms * num_rbf]
        # print(RBF_all.shape)
        offset = R_idx[:, :, None] - R_idx[:, None, :] #  [B, L] -
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.positional_embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        # print(E.shape)
        E = self.enc_edge_embedding(E)
        E = self.enc_norm_edges(E)
        

        V = torch.nn.functional.one_hot(S, 21).float()
        # print(V.shape)
        V = torch.cat([V, protein_mpnn_feat], dim=-1)#, esm_feat
        # print(V.shape)
        V = self.enc_node_embedding(V)
        V = self.enc_norm_nodes(V)


        return V, E, E_idx, X

    def features_decode(self, features, mask):
        """
        Make features for decoding. Explicit side chain atom and other atom distances.
        """

        S = features["S"]
        X = features["X"]
        X_m = features["X_m"]
        # mask = features["mask"]
        mask = mask
        E_idx = features["E_idx"]

        Y = features["Y"][:, :, : self.atom_context_num]
        Y_m = features["Y_m"][:, :, : self.atom_context_num]
        Y_t = features["Y_t"][:, :, : self.atom_context_num]

        X_m = X_m * mask[:, :, None]
        device = S.device

        B, L, _, _ = X.shape

        RBF_sidechain = []
        X_m_gathered = gather_nodes(X_m, E_idx)  # [B, L, K, 14]

        for i in range(14):
            for j in range(14):
                rbf_features = self._get_rbf(
                    X[:, :, i, :],
                    X[:, :, j, :],
                    E_idx,
                    D_mu_shape=[1, 1, 1, -1],
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    num_bins=self.num_rbf,
                )
                rbf_features = (
                    rbf_features
                    * X_m[:, :, i, None, None]
                    * X_m_gathered[:, :, :, j, None]
                )
                RBF_sidechain.append(rbf_features)

        D_XY = torch.sqrt(
            torch.sum((X[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, 14, atom_context_num]
        XY_features = self._rbf(
            D_XY,
            D_mu_shape=[1, 1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )  # [B, L, 14, atom_context_num, num_rbf]
        XY_features = XY_features * X_m[:, :, :, None, None] * Y_m[:, :, None, :, None]

        Y_t_1hot = torch.nn.functional.one_hot(
            Y_t.long(), 120
        ).float()  # [B, L, atom_context_num, 120]
        XY_Y_t = torch.cat(
            [XY_features, Y_t_1hot[:, :, None, :, :].repeat(1, 1, 14, 1, 1)], -1
        )  # [B, L, 14, atom_context_num, num_rbf+120]
        XY_Y_t = self.W_XY_project_down1(
            XY_Y_t
        )  # [B, L, 14, atom_context_num, num_rbf]
        XY_features = XY_Y_t.view([B, L, -1])

        V = self.dec_node_embedding1(XY_features)
        V = self.dec_norm_nodes1(V)

        S_1h = torch.nn.functional.one_hot(S, self.enc_node_in).float()
        S_1h_gathered = gather_nodes(S_1h, E_idx)  # [B, L, K, 21]
        S_features = torch.cat(
            [S_1h[:, :, None, :].repeat(1, 1, E_idx.shape[2], 1), S_1h_gathered], -1
        )  # [B, L, K, 42]

        F = torch.cat(
            tuple(RBF_sidechain), dim=-1
        )  # [B,L,atom_context_num,14*14*num_rbf]
        F = torch.cat([F, S_features], -1)
        F = self.dec_edge_embedding1(F)
        F = self.dec_norm_edges1(F)
        return V, F
def get_act_fxn(act: str):
    if act == 'relu':
        return F.relu
    elif act == 'gelu':
        return F.gelu
    elif act == 'elu':
        return F.elu
    elif act == 'selu':
        return F.selu
    elif act == 'celu':
        return F.celu
    elif act == 'leaky_relu':
        return F.leaky_relu
    elif act == 'prelu':
        return F.prelu
    elif act == 'silu':
        return F.silu
    elif act == 'sigmoid':
        return nn.Sigmoid()

class MLP(nn.Module):
    def __init__(self, num_in, num_inter, num_out, num_layers, act='relu', bias=True):
        super().__init__()
        
        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)
        self.W_inter = nn.ModuleList([nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)])
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)
        
        # Activation function
        self.act = get_act_fxn(act)
        
    def forward(self, X):
        
        # Embed inputs with input layer
        X = self.act(self.W_in(X))
        
        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))
            
        # Get output from output layer
        X = self.W_out(X)
        
        return X



if __name__ == "__main__":
    # test
    # test for MPNNL
    # test for MPNNL
    edge_features = 128
    node_features = 128
    num_positional_embeddings = 16
    num_chain_embeddings = 16
    num_rbf = 16
    top_k = 30
    augment_eps = 0.0
    atom37_order = False
    device = None
    atom_context_num = 16
    lower_bound = 0.0
    upper_bound = 20.0
    model = ProteinFeatures(
        edge_features=edge_features,
        node_features=node_features,
        num_positional_embeddings=num_positional_embeddings,
        num_chain_embeddings=num_chain_embeddings,
        num_rbf=num_rbf,
        top_k=top_k,
        augment_eps=augment_eps,
        atom37_order=atom37_order,
        device=device,
        atom_context_num=atom_context_num,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    features = {
        "S": torch.randn(2, 100),
        "x": torch.randn(2, 100, 5, 3),
        "mask": torch.randn(2, 100),
        "R_idx": torch.randn(2, 100),
        "chain_labels": torch.randn(2, 100),
    }
    V, E, E_idx = model.features_encode(features)
    print(V.shape, E.shape, E_idx.shape)




# %%