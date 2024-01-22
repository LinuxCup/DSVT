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
# import onnxruntime as ort
import torch.nn as nn

# from typing import Sequence, NamedTuple
import pdb
import time
import numpy as np
import torch.nn.functional as F



# for plain version, covert the torch model to onnx model

# cfg_file = "/home/zhenghu/DeepLearning/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml"
cfg_file = "/home/zhenghu/DeepLearning/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn_sparse_pre.yaml"
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
ckpt = "/home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn_sparse_pre/default/ckpt/latest_model.pth"
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False, pre_trained_path=None)
model.eval()
model.cuda()

# pdb.set_trace()

pointpillarscatter3d = model.map_to_bev_module
basebevresbackbone = model.backbone_2d
transfusion_head = model.dense_head
# shared_conv = center_head.shared_conv
# separate_heads = center_head.heads_list[0]


import matplotlib.pyplot as plt
def visualization_feature(feature):
    grid_sz_z = feature.shape[0]
    row_vis = 3
    fig, (axes) = plt.subplots((int)(grid_sz_z/row_vis + 1),row_vis)
    for i,t in enumerate(axes):
        for j,ax in enumerate(t):
            if (i*row_vis + j) >= grid_sz_z:
                continue
            im = ax.imshow(feature[i*row_vis+j])
            fig.colorbar(im, ax=ax)
    plt.show()

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


class MySortImpe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, src_coor, valid_num):
        ret = src
        ret[..., : 500] = ret[..., : 500] + src_coor[..., : 500]
        return ret

        return (src+ src_coor).long()
        

    @staticmethod
    def symbolic(g, src, src_coor, valid_num):
        return g.op("MysortPlugin",src, src_coor, valid_num)

my_sort = MySortImpe.apply

class TransfusionHead(nn.Module):
    def __init__(self, shared_conv, heatmap_head, class_encoding, decoder, prediction_head):
        super().__init__()
        self.shared_conv = shared_conv
        self.heatmap_head = heatmap_head
        self.class_encoding = class_encoding
        self.decoder = decoder
        self.prediction_head = prediction_head

        self.nms_kernel_size = 3
        self.num_proposals = 500
        self.num_classes = 6
        self.feature_map_stride = 1
        self.voxel_size = [ 0.32 * 2, 0.32 * 2, 10]
        self.point_cloud_range = [-61.44, -42.24, -5.0, 61.44, 42.24, 5.0]

        x_size = 192
        y_size = 132
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.query_radius = 10
        self.query_range = torch.arange(-self.query_radius, self.query_radius+1)
        self.query_r_coor_x, self.query_r_coor_y = torch.meshgrid(self.query_range, self.query_range) 


        self.heatmap_flatten_idx = torch.zeros([1,152064], dtype=torch.long, device='cuda')
        self.heatmap_flatten_val = torch.zeros([1,152064], device='cuda')

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        x_grid, y_grid = heatmap.shape[-2:]
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        # local_max[ :, 3, ] = F.max_pool2d(heatmap[:, 3], kernel_size=1, stride=1, padding=0)
        # local_max[ :, 4, ] = F.max_pool2d(heatmap[:, 4], kernel_size=1, stride=1, padding=0)
        # local_max[ :, 5, ] = F.max_pool2d(heatmap[:, 5], kernel_size=1, stride=1, padding=0)
        local_max[ :, 3, ] = F.max_pool2d(heatmap[:, 3].unsqueeze(0), kernel_size=1, stride=1, padding=0).squeeze(0)
        local_max[ :, 4, ] = F.max_pool2d(heatmap[:, 4].unsqueeze(0), kernel_size=1, stride=1, padding=0).squeeze(0)
        local_max[ :, 5, ] = F.max_pool2d(heatmap[:, 5].unsqueeze(0), kernel_size=1, stride=1, padding=0).squeeze(0)
        heatmap = heatmap * (heatmap == local_max)
        # pdb.set_trace()
        # visualization_feature(heatmap[0].permute(0,2,1).squeeze(dim=0).cpu())
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        
        # heatmap_flatten = heatmap.view(-1)
        # heatmap_flatten_idx = torch.nonzero(heatmap_flatten > 0.)
        # heatmap_flatten_val = torch.zeros(152064, device=heatmap_flatten.device)
        # heatmap_flatten_val_ = heatmap_flatten.index_select(dim = 0, index=heatmap_flatten_idx.squeeze(1))
        # heatmap_flatten_val[:heatmap_flatten_val_.shape[0]] = heatmap_flatten_val_
        # top_proposals = torch.zeros(500, dtype=torch.long ,device=heatmap_flatten_val.device)
        # top_proposals[..., : heatmap_flatten_idx.shape[0]] = heatmap_flatten_val[:3840].argsort(dim=-1, descending=True)[..., : heatmap_flatten_idx.shape[0]]
        # top_proposals[:heatmap_flatten_idx.shape[0]] = heatmap_flatten_idx.squeeze(1)[top_proposals[:heatmap_flatten_idx.shape[0]]]
        # top_proposals = top_proposals.view(1,-1)
        # pdb.set_trace()


        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals] # torch.Size([1, 500])
        # pdb.set_trace()


        # heatmap_flatten = heatmap.view(-1)
        # heatmap_flatten_idx = torch.zeros([1,heatmap.view(batch_size, -1).shape[-1]], dtype=torch.long, device=heatmap_flatten.device)
        # heatmap_flatten_idx_ = torch.nonzero(heatmap_flatten > 1e-3)
        # heatmap_flatten_idx[..., : heatmap_flatten_idx_.shape[0]] = heatmap_flatten_idx_.squeeze(1).view(1,-1)
        # heatmap_flatten_val = torch.zeros([1,heatmap.view(batch_size, -1).shape[-1]], device=heatmap_flatten.device)
        # heatmap_flatten_val_ = heatmap_flatten.index_select(dim = 0, index=heatmap_flatten_idx_.squeeze(1))
        # heatmap_flatten_val[..., : heatmap_flatten_val_.shape[0]] = heatmap_flatten_val_.view(1,-1)
        # top_proposals = my_sort(heatmap_flatten_val, heatmap_flatten_idx, heatmap_flatten_val_.shape[0].view(1,-1))[..., : self.num_proposals]
        # top_proposals = top_proposals.long()
        # pdb.set_trace()



        heatmap_flatten = heatmap.view(-1)
        # heatmap_flatten_idx = torch.zeros([1,heatmap.view(batch_size, -1).shape[-1]], dtype=torch.long, device=heatmap_flatten.device)
        heatmap_flatten_idx_ = torch.nonzero(heatmap_flatten > 1e-2).detach().clone()
        self.heatmap_flatten_idx[..., : 20000] = F.pad(heatmap_flatten_idx_.permute(1,0),(0,20000), 'constant',0)[..., : 20000]
        # heatmap_flatten_val = torch.zeros([1,heatmap.view(batch_size, -1).shape[-1]], device=heatmap_flatten.device)
        heatmap_flatten_val_ = heatmap_flatten.index_select(dim = 0, index=heatmap_flatten_idx_.squeeze(1)).detach().clone()
        self.heatmap_flatten_val[..., : 20000] = F.pad(heatmap_flatten_val_.view(1,-1),(0,20000), 'constant',0)[..., : 20000]
        top_proposals = my_sort(self.heatmap_flatten_val, self.heatmap_flatten_idx, heatmap_flatten_val_.shape[0].view(1,-1))[..., : self.num_proposals]
        top_proposals = top_proposals.long()
        pdb.set_trace()


        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        # pdb.set_trace()
        self.query_labels = top_proposals_class

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        # compute local key 
        top_proposals_x = top_proposals_index // x_grid # bs, num_proposals
        top_proposals_y = top_proposals_index % y_grid # bs, num_proposals
        
        # bs, num_proposal, radius * 2 + 1, radius * 2 + 1
        top_proposals_key_x = top_proposals_x[:, :, None, None] + self.query_r_coor_x[None, None, :, :].to(top_proposals.device)
        top_proposals_key_y = top_proposals_y[:, :, None, None] + self.query_r_coor_y[None, None, :, :].to(top_proposals.device)
        # bs, num_proposals, key_num
        top_proposals_key_index = top_proposals_key_x.view(batch_size, top_proposals_key_x.shape[1], -1) * x_grid + top_proposals_key_y.view(batch_size, top_proposals_key_y.shape[1], -1)
        key_mask = (top_proposals_key_index < 0) | (top_proposals_key_index >= (x_grid * y_grid))
        top_proposals_key_index = torch.clamp(top_proposals_key_index, min=0, max=x_grid * y_grid-1)
        num_proposals = top_proposals_key_index.shape[1]
        key_feat = lidar_feat_flatten.gather(index=top_proposals_key_index.view(batch_size, 1, -1).expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
        key_feat = key_feat.view(batch_size, lidar_feat_flatten.shape[1], num_proposals, -1) 
        key_pos = bev_pos.gather(index=top_proposals_key_index.view(batch_size, 1, -1).permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
        key_pos = key_pos.view(batch_size, num_proposals, -1, bev_pos.shape[-1])
        key_feat = key_feat.permute(0, 2, 1, 3).reshape(batch_size*num_proposals, lidar_feat_flatten.shape[1], -1)
        key_pos = key_pos.view(-1, key_pos.shape[2], key_pos.shape[-1])
        key_padding_mask = key_mask.view(-1, key_mask.shape[-1])

        query_feat_T = query_feat.permute(0, 2, 1).reshape(batch_size*num_proposals, -1, 1)
        query_pos_T = query_pos.view(-1, 1, query_pos.shape[-1])
        # return key_padding_mask
        # pdb.set_trace()

        query_feat_T = self.decoder(
            query_feat_T, key_feat, query_pos_T, key_pos, key_padding_mask
        )
        query_feat = query_feat_T.reshape(batch_size, num_proposals, 128).permute(0, 2, 1)

        center = self.prediction_head.center(query_feat)
        height = self.prediction_head.height(query_feat)
        dim = self.prediction_head.dim(query_feat)
        rot = self.prediction_head.rot(query_feat)
        iou = self.prediction_head.iou(query_feat)
        heatmap = self.prediction_head.heatmap(query_feat)
        center = center + query_pos.permute(0, 2, 1)
        # pdb.set_trace()
        # return torch.cat([center, height, dim, rot, iou, heatmap],dim=1)
        return center, height, dim, rot, iou, heatmap


    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values
        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        height = height - dim[:, 2:3, :] * 0.5 
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        # rot = torch.atan2(rots, rotc)
        rot = torch.atan(rots/(rotc + 1e-6))

        final_box_preds = torch.cat([center, height, dim, rot, final_preds.unsqueeze(0), final_scores.unsqueeze(0)], dim=1).permute(0, 2, 1)
        # pdb.set_trace()
        return final_box_preds


    def get_bboxes(self, center, height, dim, rot, iou, heatmap):
        score = heatmap.sigmoid()
        batch_iou = (iou + 1) * 0.5
        vel = None
        res = self.decode_bbox(score, rot, dim,
            center, height, vel,
            filter=True,)
        return res

    def forward(self, inputs):
        inputs = inputs.permute(0,1,3,2).contiguous()
        # return self.predict(inputs)
        center, height, dim, rot, iou, heatmap = self.predict(inputs)
        res = self.get_bboxes(center, height, dim, rot, iou, heatmap)

        return res

# pillarscatter,  backbone2d, centerhead
class Combine3Modules(nn.Module):
    def __init__(self, transfusionhead) -> None:
        super().__init__()
        # self.pillarscatter = pointpillarscatter3d
        self.backbone2d = basebevresbackbone
        self.transfusionhead = transfusionhead
        # self.shared_conv = shared_conv
        # self.separate_heads = separate_heads
    
    def forward(
        self, spatial_features
    ):
        # spatial_features = self.pillarscatter(voxel_features, voxel_coords)
        spatial_features = spatial_features.permute(0,3,1,2)
        spatial_features_2d = self.backbone2d(spatial_features)

        return self.transfusionhead(spatial_features_2d)
        return spatial_features_2d
        feats = self.shared_conv(spatial_features_2d)
        dense_preds = self.separate_heads(feats)
        return dense_preds

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
        opset_version=16,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        # enable_onnx_checker=False,
    )
    # pdb.set_trace()

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


    query, key, query_pos, key_pos, key_padding_mask, head_inputs = torch.load("transfusion_head_decoder.pth").values()
    head_inputs = head_inputs.permute(0,1,3,2)
    transfusion_head =  TransfusionHead(model.dense_head.shared_conv, 
                                        model.dense_head.heatmap_head, 
                                        model.dense_head.class_encoding, 
                                        model.dense_head.decoder,
                                        model.dense_head.prediction_head).eval().cuda()
    # torch.onnx.export(
    #     transfusion_head,
    #     (head_inputs),
    #     "transfusion_head.onnx", input_names=["input"],
    #     output_names=["objects"],
    #     opset_version=14
    # )





    # convert pillarscatter, backbone 2d, and center head to onnx
    jit_mode = "trace"
    input_names = ["voxel_features", "voxel_coords"]
    output_names = ['center', 'center_z', 'dim', 'rot', 'iou', 'hm']
    output_names = ['objects']
    # output_names = ['feats']
    


    base_name = "combine3modules_dynamic_shape_noscatter_zhito_2dsvt_1_2rpn"
    ts_path = f"{base_name}.ts"
    onnx_path = f"{base_name}.onnx"

    combine3modules = Combine3Modules(transfusion_head).eval().cuda()
    spatial_features = pointpillarscatter3d(voxel_features, voxel_coords)
    combine3modules_inputs = (voxel_features, voxel_coords)

    # pdb.set_trace()
    torch.onnx.export(
        combine3modules, spatial_features.permute(0,2,3,1), onnx_path,
        input_names=['input'],
        output_names=output_names,
        opset_version=14,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )


    pillars_coors_in_win_shift0 = torch.from_numpy(np.load('npy_file/dsvt_input_layer/pillars_coors_in_win_shift0.npy').reshape(-1, 3)).cuda()
    pillars_coors_in_win_shift1 = torch.from_numpy(np.load('npy_file/dsvt_input_layer/pillars_coors_in_win_shift1.npy').reshape(-1, 3)).cuda()
    # pillars_coors_in_win_shift0 = pillars_coors_in_win_shift0[:24 * 24,...]
    # pillars_coors_in_win_shift1 = pillars_coors_in_win_shift1[:24 * 24,...]
    # pdb.set_trace()
    base_name = "pos_enbedding_zhito_2dsvt_1_2rpn_dynamic"
    # base_name = "pos_embedding_zhito_2dsvt_1_2rpn"
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
