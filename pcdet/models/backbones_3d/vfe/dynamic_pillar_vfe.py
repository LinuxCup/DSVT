import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
import pdb
import numpy as np
from pcdet.ops.voxel import Voxelization, DynamicScatter


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        # pdb.set_trace()
        # np.save('./npy_file/inputs.npy', inputs.detach().cpu().numpy())
        # np.save('./npy_file/unq_inv.npy', unq_inv.type(torch.int32).detach().cpu().numpy())

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated

    # def forward(self, inputs):
    #     x = self.linear(inputs)
    #     x = self.norm(x) if self.use_norm else x
    #     x = self.relu(x)
    #     return x

class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        pdb.set_trace()

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        return features, unq_inv

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict


class DynamicPillarVFE_3d(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        # pdb.set_trace()
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]


    # @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            pdb.set_trace()
            # np.save('./npy_file/vfe/res_coors.npy', res_coors.detach().cpu().numpy())
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = torch.round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_z).type(torch.int)
        canvas_y = torch.round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_y).type(torch.int)
        canvas_x = torch.round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_x).type(torch.int)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    def forward_flat(self, batch_dict, **kwargs):
        points = batch_dict['points'][:,1:].contiguous()
        # pdb.set_trace()
        # np.save('./npy_file/points.npy', points.detach().cpu().numpy())
        voxel_dict=dict(
            voxel_size=(0.32, 0.32, 6),
            max_num_points=-1,
            point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
            max_voxels=(-1, -1)
        )
        self.voxel_layer = Voxelization(**voxel_dict)
        self.cluster_scatter = DynamicScatter(
            voxel_dict['voxel_size'], voxel_dict['point_cloud_range'], average_points=True)
        mode = 'max'
        self.vfe_scatter = DynamicScatter((0.32, 0.32, 6), (-74.88, -74.88, -2, 74.88, 74.88, 4.0),
                                          (mode != 'max'))
        voxels, coors = self.voxelize([points])

        voxel_mean, mean_coors = self.cluster_scatter(voxels, coors)
        # pdb.set_trace()
        # np.save('./npy_file/vfe/voxel_mean.npy', voxel_mean.detach().cpu().numpy())
        # np.save('./npy_file/vfe/mean_coors.npy', mean_coors.detach().cpu().numpy())



        features_ls = [points]
        points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
        f_cluster = points[:, 0:3] - points_mean[:, 0:3]
        features_ls.append(f_cluster)

        f_center = points.new_zeros(size=(points.size(0), 3))
        f_center[:, 0] = points[:, 0] - (
            coors[:, 3].type_as(points) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points[:, 1] - (
            coors[:, 2].type_as(points) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points[:, 2] - (
            coors[:, 1].type_as(points) * self.voxel_z + self.z_offset)
        features_ls.append(f_center)

        features = torch.cat(features_ls, dim=-1)

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            # pdb.set_trace()
            # np.save('./npy_file/vfe/voxel_feats.npy', voxel_feats.detach().cpu().numpy())
            # np.save('./npy_file/vfe/voxel_coors.npy', voxel_coors.detach().cpu().numpy())
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        # pdb.set_trace()
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = voxel_feats
        batch_dict['voxel_coords'] = voxel_coors

        return batch_dict

    def forward(self, batch_dict, **kwargs):
        # return self.forward_flat(batch_dict)

        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
        # pdb.set_trace()
        # np.save('./npy_file/dsvt_input_layer/points.npy', points[...,1:5].detach().cpu().numpy()) #torch.Size([91683, 4])

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz: #true
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:  #false
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        # pdb.set_trace()
        # save_vfe_onnx(self.pfn_layers, features, unq_inv)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        # pdb.set_trace()
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        # pdb.set_trace()
        # # np.save('./npy_file/dsvt_input_layer/points.npy', points[...,1:5].detach().cpu().numpy()) #torch.Size([90143, 4])
        # voxel_features_trt = torch.from_numpy(np.load('npy_file/dsvt_input_layer/voxel_feats_zhito.npy').reshape(-1, 128)).cuda()
        # voxel_coors_trt = torch.from_numpy(np.load('npy_file/dsvt_input_layer/voxel_coors_zhito.npy').reshape(-1, 4)).cuda()
        # batch_dict['pillar_features'] = batch_dict['voxel_features'] = voxel_features_trt
        # batch_dict['voxel_coords'] = voxel_coors_trt
        # pdb.set_trace()

        # pdb.set_trace()
        return batch_dict
