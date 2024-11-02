# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import to_dense_batch
import numpy as np
from torch.utils.checkpoint import checkpoint

from blocks import (
    InteractionBlock,
    MPNNL, 
    ProteinFeatures,
    MLP,
    CrossAttentionBlock,
    )
from model_utils import (
    EncLayer,
    gather_nodes,
)
from openfold.data.data_transforms import atom37_to_torsion_angles, make_atom14_masks
import openfold.np.residue_constants as rc
from openfold.utils.rigid_utils import Rigid
from data_utils import (
    get_atom14_coords,
    chi_angle_to_bin,
    nll_chi_loss,
    offset_mse,
    sc_rmsd,
    BlackHole,
    rotamer_recovery_from_coords,
)
from typing import Dict, Union, Optional
from openfold.utils.feats import atom14_to_atom37
from sc_utils import map_mpnn_to_af2_seq
class Pack(nn.Module):
    def __init__(self, 
        lig_node_dim=35,
        pro_node_dim=256,
        pro_edge_dim=256,
        hidden_dim=256,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom37_order=False,
        device="cuda",
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        num_encoder_layers=3,
        dropout=0.1,
        num_heads=4,
        predict_offset=True,
        recycle_strategy='mode',
        atten_active_fuc = 'softmax',
        n_chi_bins=72,
        n_recycle=3,
        lig_info=True,
        loss: Optional[Dict[str, Union[float, bool]]] = {
            "chi_nll_loss_weight": 1.0,
            "chi_mse_loss_weight": 1.0,
            "offset_mse_loss_weight": 1.0
        },
        ):
        super(Pack, self).__init__()
        self.n_chi_bins = n_chi_bins
        self.n_recycle = n_recycle
        self.predict_offset = predict_offset
        self.recycle_strategy = recycle_strategy
        self.loss = loss
        self.lig_info = lig_info
    
        self.lin_node_lig = nn.Sequential(
            Linear(lig_node_dim, hidden_dim),nn.LeakyReLU())

        self.conv = nn.ModuleList(
            [MPNNL(hidden_dim, hidden_dim, dropout) for _ in range(num_encoder_layers)])

        if self.lig_info:
            self.CrossAttention = CrossAttentionBlock(
                hid_dim=hidden_dim,
                dropout=dropout,
                atten_active_fuc = atten_active_fuc,
                num_heads=num_heads
                )

        self._pro_feat = ProteinFeatures(
            edge_features=pro_edge_dim,
            node_features=pro_node_dim,
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

        self.W_e = nn.Linear(pro_edge_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(pro_node_dim, hidden_dim, bias=True)
        self.W_seq = nn.Embedding(21, hidden_dim)
        self.W_recycle_SC_D_probs = nn.Linear(4 * (n_chi_bins + 1), hidden_dim)
        self.W_out_pro = nn.Linear(hidden_dim*2, hidden_dim, bias=True)
        self.W_out_chi = MLP(hidden_dim*2, hidden_dim, (n_chi_bins+1)*4, 3, act='relu')
        if predict_offset:
            self.W_out_offset = nn.Linear(hidden_dim, 4, bias=True)
                # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        # lig_scope = data['ligand_scope']
        # amino_acid_scope = data['protein_scope']
        data_l = data['ligand_features']
        data_aa = data['protein_features']

        x_l, edge_index_l, edge_feat_l,pos_l, pos_p= \
        data_l.x, data_l.edge_index, data_l.edge_attr, data_l.pos_l, data_aa.pos_p


        X,X_mask, seq,tor_an_gt,chain_label,R_idx, aa_type, ProteinMPNN_feat,esm_feat = data_aa.x, data_aa.x_mask, \
            data_aa.seq, data_aa.tor_an_gt, data_aa.chain_label, data_aa.R_idx, data_aa.aa_type, data_aa.ProteinMPNN_feat, \
            data_aa.esm_feat
        # print(X_mask.shape)
        # print("Device of x_l before lin_node_lig:", x_l.device)
        # print("Device of model weights in lin_node_lig:", self.lin_node_lig[0].weight.device)

        x_l = self.lin_node_lig(x_l)
        for conv in self.conv:
            x_l = conv(x_l, edge_index_l, edge_feat_l)
        
        x_l, mask_l = to_dense_batch(x_l, data_l.batch, fill_value=0)
        pos_l, mask_pos_l = to_dense_batch(pos_l, data_l.batch, fill_value=0)
        pos_p, mask_pos_p = to_dense_batch(pos_p, data_aa.batch, fill_value=0)
        
        
        
 
        X, mask = to_dense_batch(X, data_aa.batch, fill_value=0)
        seq,_ = to_dense_batch(seq, data_aa.batch, fill_value=0)
        aa_type,_ = to_dense_batch(aa_type, data_aa.batch, fill_value=0)
        tor_an_gt,_ = to_dense_batch(tor_an_gt, data_aa.batch,fill_value=0)
        chain_labels,_ = to_dense_batch(chain_label,data_aa.batch, fill_value=0)
        R_idx,_ = to_dense_batch(R_idx, data_aa.batch,fill_value=0)
        # print(X.shape, seq.shape, tor_an_gt.shape, chain_label.shape, R_idx.shape)
        protein_mpnn_feat, _ = to_dense_batch(ProteinMPNN_feat, data_aa.batch, fill_value=0)
        esm_feat, _ = to_dense_batch(esm_feat, data_aa.batch, fill_value=0)
        mask = mask.float() # (B, N)

        # S_af2 = torch.argmax(
        # torch.nn.functional.one_hot(seq, 21).float()
        # @ map_mpnn_to_af2_seq.to(X.device).float(),
        # -1,
        # )

        masks14_37 = make_atom14_masks({"aatype": aa_type})
        atom_14_mask = masks14_37["atom14_atom_exists"]
        atom_37_mask = masks14_37["atom37_atom_exists"]

        atom_14 = get_atom14_coords(
            X,
            seq,
            atom_14_mask,
            atom_37_mask,
            
            
            tor_an_gt,

            # outputs["final_SC_D"],
            device=X.device
        )


        feature_dict = {}
        feature_dict['x'] = atom_14
        feature_dict['mask'] = mask
        feature_dict["x_37"] = X
        feature_dict["atom37_mask"] = atom_37_mask
        feature_dict["atom14_mask"] = atom_14_mask
        feature_dict['S'] = seq
        feature_dict['tor_an_gt'] = tor_an_gt
        feature_dict['chain_labels'] = chain_labels
        feature_dict['R_idx'] = R_idx
        feature_dict['aatype'] = aa_type
        feature_dict['protein_mpnn_feat'] = protein_mpnn_feat
        feature_dict['esm_feat'] = esm_feat



        outputs, x_l = self.chi_pred(x_l, mask_l, feature_dict)
        # print(outputs["chi_probs"])
        # SC_D_mask = chi_mask
        # print(SC_D_mask.shape)
        SC_D_bin, SC_D_bin_offset = chi_angle_to_bin(feature_dict['tor_an_gt'], self.n_chi_bins, outputs["chi_mask"])
        feature_dict["SC_D_bin_ture"] = SC_D_bin
        feature_dict["SC_D_bin_offset_ture"] = SC_D_bin_offset
        chi_losses = self.chi_loss(outputs, feature_dict)
        print(chi_losses)
        return chi_losses
    
    def chi_pred(self, lig_batch, lig_mask, feature_dict):
        # Add empty previous prediction
        prevs = {
            "pred_X": torch.zeros_like(feature_dict["x"]),
            "pred_X_mask": torch.cat((
                    torch.ones_like(feature_dict["mask"][...,:5]),
                    torch.zeros_like(feature_dict["mask"][...,5:])),dim = -1),
            "pred_SC_D": torch.zeros_like(feature_dict["tor_an_gt"]),
            "pred_SC_D_probs": torch.zeros((*feature_dict["S"].shape, 4, self.n_chi_bins+1),device=feature_dict["S"].device),

        }
        with torch.no_grad():
            for _ in range(self.n_recycle):
                outputs, lig_batch = self.single_chi_pred(lig_batch, lig_mask, feature_dict, prevs)
            
                # print(out_puts)
                chi_pred = self._chi_prediction_from_probs(
                            outputs['chi_probs'],
                            outputs.get('chi_bin_offset', None),
                            strategy=self.recycle_strategy)
                aatype_chi_mask =torch.tensor(
                            rc.chi_angles_mask,
                            dtype=torch.float32,
                            device=chi_pred.device)[feature_dict["aatype"]]
                # print(aatype_chi_mask.shape)
                chi_pred = aatype_chi_mask* chi_pred
                # print(feature_dict["S"][0][1])
                # print(chi_pred[0][1])

                atom14_xyz = get_atom14_coords(
                            feature_dict["x_37"],
                            feature_dict["S"],
                            feature_dict["atom14_mask"],
                            feature_dict["atom37_mask"],
                            chi_pred,
                            device=feature_dict["x"].device)
                # print(atom14_xyz.shape)
                prevs["pred_X"] = atom14_xyz
                prevs["pred_X_mask"] = feature_dict["atom14_mask"]
                prevs["pred_SC_D"] = chi_pred
                prevs["pred_SC_D_probs"] = outputs.get("chi_probs", None)


        # final prediction
        outputs, lig_batch = self.single_chi_pred(lig_batch, lig_mask, feature_dict, prevs)
        chi_pred = self._chi_prediction_from_probs(
                    outputs['chi_probs'],
                    outputs.get('chi_bin_offset', None),
                    strategy=self.recycle_strategy)
        aatype_chi_mask =torch.tensor(
                        rc.chi_angles_mask,
                        dtype=torch.float32,
                        device=chi_pred.device)[feature_dict["aatype"]]

        chi_pred = aatype_chi_mask* chi_pred
        atom14_xyz = get_atom14_coords(
                    feature_dict["x_37"],
                    feature_dict["S"],
                    feature_dict["atom14_mask"],
                    feature_dict["atom37_mask"],
                    chi_pred,
                    device=feature_dict["x"].device)
        outputs['chi_mask'] = aatype_chi_mask
        outputs['final_SC_D'] = chi_pred
        outputs['final_X'] = atom14_xyz
        outputs["final_X_mask"] = (atom14_xyz.sum(-1) != 0).float()    
        # print(outputs.keys())


        return outputs, lig_batch

    def single_chi_pred(self, lig_batch, lig_mask, feature_dict,prevs):
        # print(feature_dict["x"][..., :4, :].shape, prevs["pred_X"][..., 4:, :].shape)
        X =torch.cat((feature_dict["x"][..., :4, :], prevs["pred_X"][..., 4:, :]), dim=-2)
        # print(prevs["pred_X"])
        S = feature_dict["S"]
        mask = feature_dict["mask"]
        feature_dict["x"] = X

        # Get the initial features
        V, E, E_idx, X = self._pro_feat.features_encode(feature_dict)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        lig_batch, h_V = self.CrossAttention(lig_batch, h_V, lig_mask, mask.float())

        h_V = h_V + self.W_recycle_SC_D_probs(prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1))
        # print(h_V.shape)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for enc_layer in self.encoder_layers:
            if torch.is_grad_enabled():
                h_V, h_E = checkpoint(enc_layer, h_V, h_E, E_idx, mask, mask_attend)

            else:
                h_V, h_E = enc_layer(h_V, h_E, E_idx, mask, mask_attend)
        out_puts = {}
        h_S = self.W_seq(S)
        # print(h_S.shape)
        h_VS = torch.cat((h_V, h_S), dim=-1)
        
        h_VS_out = self.W_out_pro(h_VS)
        # print(h_VS_out.shape,lig_batch.shape, lig_mask.shape, mask.shape)
        if self.lig_info:
            lig_batch, h_VS_out= self.CrossAttention(lig_batch, h_VS_out, lig_mask, mask.float())

        CH_logits = self.W_out_chi(h_VS).view(h_VS.shape[0], h_VS.shape[1], 4, -1)
        # print(CH_logits)
        chi_log_probs = F.log_softmax(CH_logits, dim=-1)
        # print(chi_log_probs)
        chi_probs = F.softmax(CH_logits, dim=-1)
        out_puts["chi_log_probs"] = chi_log_probs
        out_puts["chi_probs"] = chi_probs
        out_puts["chi_logits"] = CH_logits
        out_puts["V"] = h_VS_out
        if self.predict_offset:
            offset = (2*torch.pi/self.n_chi_bins) * torch.sigmoid(self.W_out_offset(h_V))
            out_puts["offset"] = offset
        return out_puts, lig_batch
    
    def _chi_prediction_from_probs(self, chi_probs, chi_bin_offset=None, strategy="mode"):        
        # One-hot encode predicted chi bin
        if strategy == "mode":
            chi_bin = torch.argmax(chi_probs, dim=-1)
        elif strategy == "sample":
            chi_bin = torch.multinomial(chi_probs.view(-1, chi_probs.shape[-1]), num_samples=1).squeeze(-1).view(*chi_probs.shape[:-1])
        chi_bin_one_hot = F.one_hot(chi_bin, num_classes=self.n_chi_bins + 1)

        # Determine actual chi value from bin
        chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / self.n_chi_bins, device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
        pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
        
        # Add bin offset if provided
        if self.predict_offset and chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / self.n_chi_bins) * torch.rand(chi_bin.shape, device=chi_bin.device)
        sampled_chi = pred_chi_bin + bin_sample_update
        
        return sampled_chi

    def sample(self, lig_batch, lig_mask, feature_dict, temperature=1.0, n_recycle=0):
        prevs = {
            "pred_X": torch.zeros_like(feature_dict["x"]),
            "pred_X_mask": torch.cat((
                    torch.ones_like(feature_dict["mask"][...,:5]),
                    torch.zeros_like(feature_dict["mask"][...,5:])),dim = -1),
            "pred_SC_D": torch.zeros_like(feature_dict["tor_an_gt"]),
            "pred_SC_D_probs": torch.zeros((*feature_dict["S"].shape, 4, self.n_chi_bins+1),device=feature_dict["S"].device),

        }

        for _ in range(n_recycle):
            sample_out, lig_batch = self.single_sample(lig_batch, lig_mask, feature_dict, prevs, temperature)
            chi_pred = self._chi_prediction_from_probs(
                sample_out['chi_probs'],
                sample_out.get('chi_bin_offset'),
                strategy=self.recycle_strategy
                )
            aatype_chi_mask =torch.tensor(
                rc.chi_angles_mask,
                dtype=torch.float32,
                device=chi_pred.device)[feature_dict["aatype"]]
            chi_pred = aatype_chi_mask * chi_pred
            atom14_xyz = get_atom14_coords(
                feature_dict["x_37"],
                feature_dict["S"],
                feature_dict["atom14_mask"],
                feature_dict["atom37_mask"],
                chi_pred,
                device=feature_dict["x"].device
            )
            prevs["pred_X"] = atom14_xyz
            prevs["pred_X_mask"] = feature_dict["atom14_mask"]
            prevs["pred_SC_D"] = chi_pred
            prevs["pred_SC_D_probs"] = sample_out.get("chi_probs", None)

            # final prediction
        sample_out, lig_batch = self.single_sample(lig_batch, lig_mask, feature_dict, prevs, temperature)

        chi_pred = self._chi_prediction_from_probs(
            sample_out['chi_probs'],
            sample_out.get('chi_bin_offset'),
            strategy=self.recycle_strategy
            )
        aatype_chi_mask =torch.tensor(
            rc.chi_angles_mask,
            dtype=torch.float32,
            device=chi_pred.device)[feature_dict["aatype"]]
        chi_pred = aatype_chi_mask * chi_pred
        atom14_xyz = get_atom14_coords(
            feature_dict["x_37"],
            feature_dict["S"],
            feature_dict["atom14_mask"],
            feature_dict["atom37_mask"],
            chi_pred,
            device=feature_dict["x"].device
        )

        sample_out['chi_mask'] = aatype_chi_mask
        sample_out['final_SC_D'] = chi_pred
        sample_out['final_X'] = atom14_xyz

        return sample_out

    def single_sample(self, lig_batch, lig_mask, feature_dict, prevs, temperature=1.0):
        X =torch.cat((feature_dict["x"][..., :4, :], prevs["pred_X"][..., 4:, :]), dim=-2)
        S = feature_dict["S"]
        mask = feature_dict["mask"]
        feature_dict["x"] = X

        # Get the initial features
        V, E, E_idx, X = self._pro_feat.features_encode(feature_dict)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        lig_batch, h_V = self.CrossAttention(lig_batch, h_V, lig_mask, mask.float())
        h_V = h_V + self.W_recycle_SC_D_probs(prevs["pred_SC_D_probs"].view(*prevs["pred_SC_D_probs"].shape[:-2], -1))

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for enc_layer in self.encoder_layers:
            if torch.is_grad_enabled():
                h_V, h_E = checkpoint(enc_layer, h_V, h_E, E_idx, mask, mask_attend)

            else:
                h_V, h_E = enc_layer(h_V, h_E, E_idx, mask, mask_attend)
        out_puts = {}
        h_S = self.W_seq(S)
        h_VS = torch.cat((h_V, h_S), dim=-1)
        
        h_VS_out = self.W_out_pro(h_VS)
        if self.lig_info:
        
            lig_batch, h_VS_out= self.CrossAttention(lig_batch, h_VS_out, lig_mask, mask.float())
        if temperature >0.0:
            CH_logits = self.W_out_chi(h_VS).view(h_VS.shape[0], h_VS.shape[1], 4, -1) / temperature
            
            chi_probs = F.softmax(CH_logits, dim=-1)
            CH = torch.multinomial(chi_probs.view(-1, chi_probs.shape[-1]), num_samples=1).view(CH_logits.shape[0], CH_logits.shape[2], -1).squeeze(-1)
        else:
            CH_logits = self.W_out_chi(h_VS).view(h_VS.shape[0], h_VS.shape[1], 4, -1)
            chi_probs = F.softmax(CH_logits, dim=-1)
            CH = torch.argmax(chi_probs, dim=-1)   

        if self.predict_offset:
            offset = (2*torch.pi/self.n_chi_bins) * torch.sigmoid(self.W_out_offset(h_V))
        out_put = {
            "chi_bin": CH,
            "chi_probs": chi_probs,
            "chi_bin_offset": offset,
            "chi_logits": CH_logits
        }
            
        return out_put, lig_batch
    
    def chi_loss(self, outputs, feature_dict,use_sc_bf_mask=False, _return_breakdown=False, _logger=BlackHole(), _log_prefix="train"):
        if use_sc_bf_mask:
            mask = feature_dict["sc_bf_mask"]
        else:
            SC_D_mask = outputs["chi_mask"]
    


        loss_fns = {
            "rmsd_loss": lambda: sc_rmsd(
                outputs["final_X"],
                feature_dict["x"], 
                feature_dict["aatype"],
                feature_dict["atom14_mask"],
                feature_dict["mask"],
                _metric = _logger.get_metric(_log_prefix + "rmsd")),

            "rotamer_recovery": lambda: rotamer_recovery_from_coords(
                feature_dict["aatype"],
                torch.atan2(feature_dict['tor_an_gt'][..., 0], feature_dict['tor_an_gt'][..., 1]),
                outputs["final_X"],
                feature_dict["mask"],
                outputs["chi_mask"],
                _metric = _logger.get_metric(_log_prefix + "rotamer_recovery")),

            "chi_nll_loss": lambda: nll_chi_loss(
                outputs["chi_log_probs"],
                feature_dict["SC_D_bin_ture"],
                feature_dict["aatype"],
                outputs["chi_mask"],
                _metric = _logger.get_metric(_log_prefix + "chi_nll_loss")),

            "offset_mse_loss": lambda: offset_mse(
                outputs["offset"],
                feature_dict["SC_D_bin_offset_ture"],
                outputs["chi_mask"],
                _metric = _logger.get_metric(_log_prefix + "chi_mse_loss"))}
        total_loss = 0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss.get(loss_name + "_weight", 0.0)
            loss = loss_fn()
            # print(loss)
            if (torch.isnan(loss) or torch.isinf(loss)):
                self.log.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            total_loss += weight * loss
            losses[loss_name] = loss.detach().cpu().clone()
        if _return_breakdown:
            return total_loss, losses
        return total_loss, losses
        

if __name__ == '__main__':
    import os
    import pandas as pd
    from dataset_pack import GraphDataset, PLIDataLoader
    data_root = '../PLmodel/supervised/data/pdbbind/'
    data_df = pd.read_csv(os.path.join(data_root, 'data.csv')).head(100)
    
    # # three hours
    toy_set = GraphDataset(data_root, data_df, graph_type='MPNN', dis_threshold=10, create=False)
    toy_loader = PLIDataLoader(toy_set, batch_size=64, shuffle=True)
    model = Pack()
    for data in toy_loader:
       loss = model(data)
      
    #    print( loss, out.shape)