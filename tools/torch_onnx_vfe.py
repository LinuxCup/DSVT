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

class ScatterMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,src,src_channel_max,index):
        # 调unique仅为了输出对应的维度信息
        # src =  torch.cat((src, index.unsqueeze(axis=1)), 1)
        temp = torch.unique(index)
        out = torch.zeros((temp.shape[0],src.shape[1]),dtype=torch.float32,device=src.device)
        out = torch.cat((out, out), dim=0)
        return out.unsqueeze(0)
    
    @staticmethod
    def symbolic(g, src,src_channel_max,index):
        return g.op("ScatterMaxPlugin",src,src_channel_max,index)

scatter_max = ScatterMax.apply

class VFETRT(nn.Module):
    def __init__(self, pfn_layers):
        super().__init__()
        self.pfn_layers = pfn_layers

    def forward(
        self,
        inputs,
        unq_inv
    ):
        unq_inv = unq_inv.squeeze(0)
        num_voxels = unq_inv.max() + 1

        x = self.pfn_layers[0].linear(inputs)
        x = self.pfn_layers[0].norm(x)
        x = self.pfn_layers[0].relu(x)
        x_max = scatter_max(x, x.max(dim=1)[0], unq_inv.type(torch.int32)).squeeze(0)
        pdb.set_trace()
        # x_max = torch_scatter.scatter_max(x, unq_inv.type(torch.long), dim=0)[0]
        # x_max = x_max[:num_voxels,]
        x = torch.cat([x, x_max[unq_inv.type(torch.long), :]], dim=1)

        x = self.pfn_layers[1].linear(x)
        x = self.pfn_layers[1].norm(x)
        x = self.pfn_layers[1].relu(x)
        x_max = scatter_max(x, x.max(dim=1)[0], unq_inv.type(torch.int32)).squeeze(0)
        # x_max = torch_scatter.scatter_max(x, unq_inv.type(torch.long), dim=0)[0]
        # x_max = x_max[:num_voxels,]

        # pdb.set_trace()
        return x_max

# batch_dict = torch.load("batch_dict.pth", map_location="cuda")
# inputs, unq_inv = model.vfe(batch_dict)
# pdb.set_trace()

with torch.no_grad():

    inputs = torch.from_numpy(np.load('npy_file/inputs.npy')).cuda()  # torch.Size([164883, 11])
    unq_inv = torch.from_numpy(np.load('npy_file/unq_inv.npy')).cuda().unsqueeze(0).type(torch.int32) #torch.Size([164883])
    # pdb.set_trace()
    # inputs = torch.zeros((200000, 11),dtype=torch.float32,device='cuda:0')
    # unq_inv = torch.zeros((200000),dtype=torch.int64,device='cuda:0')
    # inputs[:164883,] = inputs_data
    # unq_inv[:164883] = unq_inv_data
    num_points = torch.tensor(164883).unsqueeze(dim=0).cuda()
    num_voxels = torch.tensor(20553).unsqueeze(dim=0).unsqueeze(dim=0).cuda()  # unq_inv.max().unsqueeze(dim=0)
    # pdb.set_trace()
    vfetrt_input = (inputs, unq_inv)

    jit_mode = "trace"
    input_names = [
        'inputs',
        'unq_inv',
        # 'num_points',
        # 'num_voxels'
    ]
    output_names = ["output",]


    dynamic_axes = {
        "inputs": {
            0: "points_number",
        },
        "unq_inv": {
            1: "points_number",
        },
        # "num_points": {
        #     1: "contant1"
        # },
        # "num_voxels": {
        #     1: "contant1"
        # }
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
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
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
