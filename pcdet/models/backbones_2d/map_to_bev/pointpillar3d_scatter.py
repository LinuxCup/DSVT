import torch
import torch.nn as nn


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, pillar_features, coords, **kwargs):
    # def forward(self, batch_dict, **kwargs):
        # pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        # import pdb
        # pdb.set_trace()
        batch_size = 1
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
        # import pdb
        # pdb.set_trace()
        # import numpy as np
        # np.save('./npy_file/batch_spatial_features.npy', batch_spatial_features.detach().cpu().numpy())
        # torch.Size([1, 128, 468, 468])
        return batch_spatial_features
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
