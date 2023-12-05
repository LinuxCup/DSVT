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
import numpy as np



# for plain version, covert the torch model to onnx model

# cfg_file = "/home/zhenghu/DeepLearning/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml"
cfg_file = "/home/zhenghu/DeepLearning/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn.yaml"
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
# ckpt = "/home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth"
ckpt = "/home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn/default/ckpt/latest_model.pth"
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False, pre_trained_path=None)
model.eval()
model.cuda()

# pdb.set_trace()

pointpillarscatter3d = model.map_to_bev_module
basebevresbackbone = model.backbone_2d
center_head = model.dense_head
shared_conv = center_head.shared_conv
separate_heads = center_head.heads_list[0]




# backbone3d
class AllPtransBlocksTRT(nn.Module):
    def __init__(self, ptransblocks_list, layer_norms_list):
        super().__init__()
        self.layer_norms_list = layer_norms_list
        self.ptransblock_list = ptransblocks_list
    def forward(
        self,
        pillar_features,
        set_voxel_inds_tensor_shift_0,
        set_voxel_inds_tensor_shift_1,
        set_voxel_masks_tensor_shift_0,
        set_voxel_masks_tensor_shift_1,
        pos_embed_tensor,
        # voxel_number,
        # set_number_shift_0,
        # set_number_shift_1
    ):
        # pdb.set_trace()
        # pillar_features = pillar_features[:voxel_number[0]]
        # set_voxel_inds_tensor_shift_0 = set_voxel_inds_tensor_shift_0[:,:set_number_shift_0[0]]
        # set_voxel_inds_tensor_shift_1 = set_voxel_inds_tensor_shift_1[:,:set_number_shift_1[0]]
        # set_voxel_masks_tensor_shift_0 = set_voxel_masks_tensor_shift_0[:,:set_number_shift_0[0]]
        # set_voxel_masks_tensor_shift_1 = set_voxel_masks_tensor_shift_1[:,:set_number_shift_1[0]]
        outputs = pillar_features

        residual = outputs
        blc_id = 0
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 1
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)
        return outputs

        residual = outputs
        blc_id = 2
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        residual = outputs
        blc_id = 3
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed, True)
        outputs = self.ptransblock_list[blc_id].encoder_list[set_id](*inputs)
        
        outputs = self.layer_norms_list[blc_id](residual + outputs)

        return outputs


# pillarscatter,  backbone2d, centerhead
class Combine3Modules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pillarscatter = pointpillarscatter3d
        self.backbone2d = basebevresbackbone
        self.shared_conv = shared_conv
        self.separate_heads = separate_heads
    
    def forward(
        self, spatial_features
    ):
        # spatial_features = self.pillarscatter(voxel_features, voxel_coords)
        spatial_features = spatial_features.permute(0,3,1,2)
        spatial_features_2d = self.backbone2d(spatial_features)
        feats = self.shared_conv(spatial_features_2d)
        dense_preds = self.separate_heads(feats)
        return dense_preds


class PosEnbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pillarscatter = pointpillarscatter3d
        self.posembed_layers = model.backbone_3d.input_layer.posembed_layers
        self.window_shape = cfg['MODEL']['BACKBONE_3D']['INPUT_LAYER']['window_shape'][0]
        # pdb.set_trace()
    
    def forward(
        self, coors_in_win_shift0, coors_in_win_shift1
    ):
        pos_total = []
        for block_id in range(2):
            pos_bolck = []

            window_shape = self.window_shape
            coors_in_win = coors_in_win_shift0
            embed_layer = self.posembed_layers[0][block_id][0]     
            location = torch.stack([coors_in_win[...,2], coors_in_win[...,1]], dim=1) - 6.
            pos_embed = embed_layer(location)
            pos_bolck.append(pos_embed)

            window_shape = self.window_shape
            coors_in_win = coors_in_win_shift1
            embed_layer = self.posembed_layers[0][block_id][0]
            location = torch.stack([coors_in_win[...,2], coors_in_win[...,1]], dim=1) - 12.
            pos_embed = embed_layer(location)
            pos_bolck.append(pos_embed)


            pos_total.append(torch.stack(pos_bolck))
            # pdb.set_trace()
        return torch.stack(pos_total)



# batch_dict = torch.load("batch_dict_waymo.pth", map_location="cuda")
batch_dict = torch.load("batch_dict_zhito.pth", map_location="cuda")
# points = batch_dict["points"]
points = torch.load('batch_dict_waymo.pth')
inputs = points

with torch.no_grad():
    ptranshierarchy3d = model.backbone_3d
    # plain version, just one stage
    ptransblocks_list = ptranshierarchy3d.stage_0
    layer_norms_list = ptranshierarchy3d.residual_norm_stage_0

    batch_dict = model.vfe(batch_dict)
    pillar_features, voxel_coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
    voxel_features = model.backbone_3d(batch_dict)['voxel_features']

    voxel_info = ptranshierarchy3d.input_layer(batch_dict)
    set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(2)] for s in range(1)]
    set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(2)] for s in range(1)]
    pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(2)] for b in range(2)] for s in range(1)]

    # pdb.set_trace()
    allptransblockstrt_inputs = (
        pillar_features,
        set_voxel_inds_list[0][0],
        set_voxel_inds_list[0][1],
        set_voxel_masks_list[0][0],
        set_voxel_masks_list[0][1],
        torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0),
        # torch.tensor([11577]),
        # torch.tensor([585]),
        # torch.tensor([399])
    )
    # import torch.nn.functional as F
    # allptransblockstrt_inputs = (
    #     F.pad(pillar_features, pad=(0,0,0,25000-11577)),
    #     F.pad(set_voxel_inds_list[0][0], pad=(0,0,0,1200-585,0,0)),
    #     F.pad(set_voxel_inds_list[0][1], pad=(0,0,0,1000-399,0,0)),
    #     F.pad(set_voxel_masks_list[0][0], pad=(0,0,0,1200-585,0,0)),
    #     F.pad(set_voxel_masks_list[0][1], pad=(0,0,0,1000-399,0,0)),
    #     F.pad(torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0), pad=(0,0,0,25000-11577)),
    #     F.pad(torch.ones(11577).cuda(), pad=(0,25000-11577)),
    #     F.pad(torch.ones(585).cuda(), pad=(0,1200-585)),
    #     F.pad(torch.ones(399).cuda(), pad=(0,1000-399))
    # )
    # test_index = torch.arange(0,10)


    jit_mode = "trace"
    input_names = [
        'src',
        'set_voxel_inds_tensor_shift_0', 
        'set_voxel_inds_tensor_shift_1', 
        'set_voxel_masks_tensor_shift_0', 
        'set_voxel_masks_tensor_shift_1',
        'pos_embed_tensor',
        # 'voxel_number',
        # 'set_number_shift_0',
        # 'set_number_shift_1'
    ]
    output_names = ["output",]

    dynamic_axes = {
        "src": {
            0: "voxel_number",
        },
        "set_voxel_inds_tensor_shift_0": {
            1: "set_number_shift_0",
        },
        "set_voxel_inds_tensor_shift_1": {
            1: "set_number_shift_1",
        },
        "set_voxel_masks_tensor_shift_0": {
            1: "set_number_shift_0",
        },
        "set_voxel_masks_tensor_shift_1": {
            1: "set_number_shift_1",
        },
        "pos_embed_tensor": {
            2: "voxel_number",
        },
        "output": {
            0: "voxel_number",
        }
    }

    base_name = "dsvt_blocks_zhito_2dsvt_1_2rpn"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"


    # convert backbone3d to onnx
    allptransblocktrt = AllPtransBlocksTRT(ptransblocks_list, layer_norms_list).eval().cuda()
    torch.onnx.export(
        allptransblocktrt,
        allptransblockstrt_inputs,
        onnx_path, input_names=input_names,
        output_names=output_names,dynamic_axes=dynamic_axes,
        opset_version=14,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        # enable_onnx_checker=False,
    )
    pdb.set_trace()

    # # test onnx
    # ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'], verbose = True)
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(pillar_features),
    #               ort_session.get_inputs()[1].name: to_numpy(set_voxel_inds_list[0][0]),
    #               ort_session.get_inputs()[2].name: to_numpy(set_voxel_inds_list[0][1]),
    #               ort_session.get_inputs()[3].name: to_numpy(set_voxel_masks_list[0][0]),
    #               ort_session.get_inputs()[4].name: to_numpy(set_voxel_masks_list[0][1]),
    #               ort_session.get_inputs()[5].name: to_numpy(torch.stack([torch.stack(v, dim=0) for v in pos_embed_list[0]], dim=0)),}
    # ort_outs = ort_session.run(None, ort_inputs)[0]
    # # pdb.set_trace()


    # convert pillarscatter, backbone 2d, and center head to onnx
    jit_mode = "trace"
    input_names = ["voxel_features", "voxel_coords"]
    output_names = ['center', 'center_z', 'dim', 'rot', 'iou', 'hm']


    base_name = "combine3modules_dynamic_shape_noscatter_zhito_2dsvt_1_2rpn"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"

    combine3modules = Combine3Modules().eval().cuda()
    spatial_features = pointpillarscatter3d(voxel_features, voxel_coords)
    combine3modules_inputs = (voxel_features, voxel_coords)

    pdb.set_trace()
    torch.onnx.export(
        combine3modules, spatial_features.permute(0,2,3,1), onnx_path,
        input_names=['input'],
        output_names=output_names,
        opset_version=14,
    )


    pillars_coors_in_win_shift0 = torch.from_numpy(np.load('npy_file/dsvt_input_layer/pillars_coors_in_win_shift0.npy').reshape(-1, 3)).cuda()
    pillars_coors_in_win_shift1 = torch.from_numpy(np.load('npy_file/dsvt_input_layer/pillars_coors_in_win_shift1.npy').reshape(-1, 3)).cuda()
    pillars_coors_in_win_shift0 = pillars_coors_in_win_shift0[:24 * 24,...]
    pillars_coors_in_win_shift1 = pillars_coors_in_win_shift1[:24 * 24,...]
    pdb.set_trace()
    # base_name = "pos_enbedding_zhito_dynamic"
    base_name = "pos_embedding_zhito_2dsvt_1_2rpn"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"

    pos_enbendding_inputs = (pillars_coors_in_win_shift0, pillars_coors_in_win_shift1)
    input_names = ["coors_in_win_shift0", "coors_in_win_shift1"]
    pos_enbendding = PosEnbedding().eval().cuda()
    output_names = ["output",]
    dynamic_axes = {
        "coors_in_win_shift0": {
            0: "voxel_number",
        },
        "coors_in_win_shift1": {
            0: "voxel_number",
        },
        "output": {
            2: "voxel_number",
        }
    }
    torch.onnx.export(
        pos_enbendding,
        pos_enbendding_inputs,
        onnx_path, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
        # output_names=output_names,
        opset_version=14,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        # enable_onnx_checker=False,
    )



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
