import torch
import torch.nn as nn
import numpy as np
import pdb
from pcdet.models.dense_heads.detr_head import DETRHead


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz
        self.detr_head = self.model_cfg.get('DENSE_HEAD', None)
        if self.detr_head:
            self.detr_head = DETRHead(model_cfg.DENSE_HEAD, self.detr_head.HIDDEN_CHANNEL, self.detr_head.NUM_HEADS, self.detr_head.FFN_CHANNEL,
                                 self.detr_head.DROPOUT, self.detr_head.ACTIVATION)

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        if self.detr_head:
            batch_dict = self.detr_head(batch_dict)
            return batch_dict
        # pdb.set_trace()
        # feat_trt = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/inference_framework/combine_inputs.npy').reshape(1,264,384,128)).permute(0,3,1,2).cuda()
        # batch_dict['spatial_features'] = feat_trt

        return batch_dict
