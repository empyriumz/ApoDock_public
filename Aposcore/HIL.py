
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from gvp.model import GVP, GVPConvLayer, LayerNorm 

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



class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1, plm=True):

        super(GVP_embedding, self).__init__()
        
        # nn.Embedding(20,20)
        if seq_in and plm:
            self.W_s = nn.Embedding(20, 1280)
            node_in_dim = (node_in_dim[0] + 1280, node_in_dim[1])
        elif seq_in and not plm:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        self.plm = plm

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if self.plm:
            seq = seq
            # print('plm')
            # print('seq',np.array(seq).shape)
        else:
            seq = self.W_s(seq)
            # print('without plm')

        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        # print('pass!')

        return out


# class AttentionBlock(nn.Module):
#     def __init__(self, hid_dim, dropout):
#         super(AttentionBlock, self).__init__()
#         self.f_q = nn.Linear(hid_dim, hid_dim)
#         self.f_k = nn.Linear(hid_dim, hid_dim)
#         self.f_v = nn.Linear(hid_dim, hid_dim)
#         self.hid_dim = hid_dim
#         self.sqrt_dk = torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query, key, value, mask=None):
#         q = self.f_q(query)  # [B, N_q, F]
#         k = self.f_k(key).transpose(-2, -1)  # [B, F, N_k]
#         v = self.f_v(value)  # [B, N_k, F]

#         # Apply dot product attention and scale the scores
#         scores = torch.matmul(q, k) / self.sqrt_dk

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         attention_weights = F.softmax(scores,dim=-1)  # [B, N_q, N_k]
#         # attention_weights = self.dropout(attention_weights)

#         # Weighted sum of values
#         output = torch.matmul(attention_weights, v)
#         return output
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
        ligand_features = self.linear_lig(lig_att)+ ligand_features #

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

        # self.side_chain_angle = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hid_dim, 4*2), # 4 side chain angles for each amino acid(sine and cosine)
        #     nn.LeakyReLU(),            
        # )

    def forward(self, lig_features, aa_features, lig_mask=None, aa_mask=None):
            # Apply transformations to ligand and amino acid features
            if self.crossAttention:
                lig_features, aa_features = self.prot_lig_attention(lig_features, aa_features, lig_mask, aa_mask)
            
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
            return interaction, interaction_mask#, angles
    
class Intrablock(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Intrablock, self).__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

    def forward(self, x_l, edge_index_l, mask_l=None):
        # Compute interaction features
        x_l_1 = x_l.unsqueeze(2)
        x_l_2 = x_l.unsqueeze(1)
        interaction = torch.mul(x_l_1, x_l_2)

        # Remove interactions corresponding to three consecutive covalent bonds, make index 0
        removed_covalent_index = []

        if mask_l is not None:
            for i in range(edge_index_l.size(1)):
                src, dst = edge_index_l[0, i], edge_index_l[1, i]
                if dst - src == 3:
                    if src < interaction.shape[1] and dst < interaction.shape[2]:
                        interaction[:, src, dst] = 0
                        interaction[:, dst, src] = 0
                        removed_covalent_index.extend([src.item(), dst.item()])
                        # Update mask_l to mask interactions between atoms separated by three consecutive covalent bonds
                        mask_l[:, src] = 0
                        mask_l[:, dst] = 0

        # Apply mask if provided
        if mask_l is not None:
            mask_l_1 = mask_l.unsqueeze(1)
            mask_l_2 = mask_l.unsqueeze(2)
            mask_l = mask_l_1 & mask_l_2
            mask_l = mask_l & ~(torch.eye(mask_l.shape[1], dtype=torch.bool, device=mask_l.device))
            interaction = interaction[mask_l]

        # Apply MLP to interaction features
        interaction = self.mlp(interaction)

        return interaction, mask_l



# %%