import os
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
import datetime
# import tensorrt as trt
# import numpy as np
import torch
# import onnx
import onnxruntime as ort
import torch.nn as nn

# from typing import Sequence, NamedTuple
import pdb
import time
import torch_scatter
import numpy as np
import torch.nn.functional as F
from pcdet.ops.voxel import Voxelization, DynamicScatter

torch.backends.cudnn.enable =True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# for plain version, covert the torch model to onnx model

cfg_file = "/home/zhenghu/DeepLearning/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml"
cfg_from_yaml_file(cfg_file, cfg)

log_file = 'logs/log_trt_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
logger = common_utils.create_logger(log_file, rank=0)
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=8, logger=logger, training=False
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
ckpt = "/home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth"
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False, pre_trained_path=None)
model.eval()
model.cuda()


pointpillarscatter3d = model.map_to_bev_module
basebevresbackbone = model.backbone_2d
center_head = model.dense_head
shared_conv = center_head.shared_conv
separate_heads = center_head.heads_list[0]
vfe = model.vfe

# class ScatterMax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,src,src_channel_max,index):
#         # 调unique仅为了输出对应的维度信息
#         # src =  torch.cat((src, index.unsqueeze(axis=1)), 1)
#         temp = torch.unique(index)
#         out = torch.zeros((temp.shape[0],src.shape[1]),dtype=torch.float32,device=src.device)
#         out = torch.cat((out, out), dim=0)
#         return out.unsqueeze(0)
    
#     @staticmethod
#     def symbolic(g, src,src_channel_max,index):
#         return g.op("ScatterMaxPlugin",src,src_channel_max,index)

# scatter_max = ScatterMax.apply

class VoxelizationImpe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,src):

        voxel_dict=dict(
            voxel_size=(0.32, 0.32, 6),
            max_num_points=-1,
            point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
            max_voxels=(-1, -1)
        )
        voxel_layer = Voxelization(**voxel_dict)
        return voxel_layer(src)

        # res_coors = torch.from_numpy(np.load('npy_file/vfe/res_coors.npy')).cuda()
        res_coors = torch.zeros((167070, 3), dtype = torch.int32).cuda()
        pdb.set_trace()
        return res_coors
    
    @staticmethod
    def symbolic(g, src):
        return g.op("VoxelizationPlugin",src)

voxelization = VoxelizationImpe.apply


class DynamicScatterCoordImpe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,voxels, coors):
        voxel_dict=dict(
            voxel_size=(0.32, 0.32, 6),
            max_num_points=-1,
            point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
            max_voxels=(-1, -1)
        )
        cluster_scatter = DynamicScatter(voxel_dict['voxel_size'], voxel_dict['point_cloud_range'], True)
        voxel_mean, mean_coors = cluster_scatter(voxels, coors)
        # mean_coors = torch.zeros((21000, 4), dtype = torch.int32).cuda()
        return mean_coors

    
    @staticmethod
    def symbolic(g, voxels, coors):
        return g.op("DynamicScatterCoordPlugin",voxels, coors)

dynamic_scatter_coord = DynamicScatterCoordImpe.apply



class DynamicScatterImpe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,voxels, coors, reduce_type):
        voxel_dict=dict(
            voxel_size=(0.32, 0.32, 6),
            max_num_points=-1,
            point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
            max_voxels=(-1, -1)
        )
        cluster_scatter = DynamicScatter(voxel_dict['voxel_size'], voxel_dict['point_cloud_range'], True)
        vfe_scatter = DynamicScatter(voxel_dict['voxel_size'], voxel_dict['point_cloud_range'],False)

        if reduce_type.item() is True:
            voxel_mean, mean_coors = cluster_scatter(voxels, coors)
            return voxel_mean
            return torch.cat((voxel_mean, mean_coors), dim=1)
        else:
            voxel_feats, voxel_coors = vfe_scatter(voxels, coors)
            return voxel_feats
            return torch.cat((voxel_feats, voxel_coors), dim=1)



        ctx.reduce_type = reduce_type



        # voxel_mean = torch.from_numpy(np.load('npy_file/vfe/voxel_mean.npy')).cuda()
        # mean_coors = torch.from_numpy(np.load('npy_file/vfe/mean_coors.npy')).cuda()
        # voxel_feats = torch.from_numpy(np.load('npy_file/vfe/voxel_feats.npy')).cuda()
        # voxel_coors = torch.from_numpy(np.load('npy_file/vfe/voxel_coors.npy')).cuda()
        voxel_mean = torch.zeros((21000, 5), dtype = torch.float).cuda()
        mean_coors = torch.zeros((21000, 4), dtype = torch.int32).cuda()
        voxel_feats = torch.zeros((21000, 64), dtype = torch.float).cuda()
        voxel_coors = torch.zeros((21000, 4), dtype = torch.int32).cuda()
        # pdb.set_trace()

        # pdb.set_trace()
        if reduce_type.item() is True:
            # return torch.cat((voxel_mean, mean_coors), dim=1)
            return voxel_mean
        else:
            # return torch.cat((voxel_feats, voxel_coors), dim=1)
            return voxel_feats
    
    @staticmethod
    def symbolic(g, voxels, coors, reduce_type):
        return g.op("DynamicScatterPlugin",voxels, coors, reduce_type)

dynamic_scatter = DynamicScatterImpe.apply


class VFETRT(nn.Module):
    def __init__(self, pfn_layers):
        super().__init__()
        self.pfn_layers = pfn_layers
        voxel_dict=dict(
            voxel_size=(0.32, 0.32, 6),
            max_num_points=-1,
            point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
            max_voxels=(-1, -1)
        )
        self.point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0]
        self.voxel_x, self.voxel_y, self.voxel_z = 0.32, 0.32, 6
        # self.voxel_layer = Voxelization(**voxel_dict)
        # self.cluster_scatter = DynamicScatter(
        #     voxel_dict['voxel_size'], voxel_dict['point_cloud_range'], average_points=True)
        # mode = 'max'
        # self.vfe_scatter = DynamicScatter((0.32, 0.32, 6), (-74.88, -74.88, -2, 74.88, 74.88, 4.0),
        #                                   (mode != 'max'))
        
        self.x_offset = voxel_dict['voxel_size'][0] / 2 + voxel_dict['point_cloud_range'][0]
        self.y_offset = voxel_dict['voxel_size'][1] / 2 + voxel_dict['point_cloud_range'][1]
        self.z_offset = voxel_dict['voxel_size'][2] / 2 + voxel_dict['point_cloud_range'][2]

    def forward(
        self,
        points
    ):
        points, coors = self.voxelize([points])
        pdb.set_trace()

        # voxel_mean, mean_coors = self.cluster_scatter(voxels, coors)
        # pdb.set_trace()
        voxel_coors = dynamic_scatter_coord(points, coors)
        average_points = torch.tensor([True])
        voxel_mean = dynamic_scatter(points, coors, average_points)
        mean_coors = voxel_coors
        # pdb.set_trace()
        # (voxel_mean, mean_coors) = res
        # res = dynamic_scatter(points, coors, average_points)
        # voxel_mean = res[:,:5]
        # mean_coors = res[:,-4:].type(torch.int32)
        # pdb.set_trace()
        

        features_ls = [points]
        points_mean = self.map_voxel_center_to_point(
                coors.detach().clone(), voxel_mean.detach().clone(), mean_coors.detach().clone())
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
            # voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            average_points = torch.tensor([False])
            voxel_feats = dynamic_scatter(point_feats, coors, average_points)
            # pdb.set_trace()
            # res = dynamic_scatter(point_feats, coors, average_points)
            # voxel_feats = res[:,:64]
            # voxel_coors = res[:,-4:].type(torch.int32)
            # voxel_feats = torch.zeros((21000, 64), dtype = torch.float).cuda()
            # voxel_coors = torch.zeros((21000, 4), dtype = torch.int32).cuda()
            # pdb.set_trace()

            if i != len(self.pfn_layers) - 1:
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        # return res
        return voxel_feats, voxel_coors

    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        for res in points:
            # res_coors = self.voxel_layer(res)
            res_coors = voxelization(res)
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
        # self.point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0]
        # self.voxel_x, self.voxel_y, self.voxel_z = 0.32, 0.32, 6
        pdb.set_trace()
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_z)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_y)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_x)
        # canvas_channel = voxel_mean.size(1)
        # batch_size = pts_coors[-1, 0].int() + 1
        batch_size = 1
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

# batch_dict = torch.load("batch_dict.pth", map_location="cuda")
# inputs, unq_inv = model.vfe(batch_dict)
# pdb.set_trace()

with torch.no_grad():

    points = torch.from_numpy(np.load('npy_file/points.npy')).cuda()
    # pdb.set_trace()
    vfetrt_input = (points)

    jit_mode = "trace"
    input_names = [
        'points'
    ]
    output_names = ["voxel_feats", "voxel_coors"]


    dynamic_axes = {
        "points": {
            0: "points_number",
        }
    }

    base_name = "vfe_trt"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"

    vfetrt = VFETRT(model.vfe.pfn_layers).eval().cuda()

    torch.onnx.export(
        vfetrt,
        vfetrt_input,
        onnx_path, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
        opset_version=14,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        verbose=True,
        # enable_onnx_checker=False
    )
    # pdb.set_trace()


'''
    conver onnx to trt engine, example of combine3modules_dynamic_shape
'''
# trtexec --onnx=combine3modules_dynamic_shape.onnx \
# --saveEngine=combine3modules_dynamic_shape.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=1 --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x192,voxel_coords:3000x4 --optShapes=voxel_features:25000x192,voxel_coords:25000x4 --maxShapes=voxel_features:35000x192,voxel_coords:35000x4 > debug.log 2>&1

# fp16
# trtexec --onnx=combine3modules_dynamic_shape.onnx --saveEngine=combine3modules_dynamic_shape_fp16.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=1 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x192,voxel_coords:3000x4 --optShapes=voxel_features:25000x192,voxel_coords:25000x4 --maxShapes=voxel_features:35000x192,voxel_coords:35000x4 > debug.log 2>&1
