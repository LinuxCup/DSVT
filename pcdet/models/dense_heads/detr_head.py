import copy
from typing import Optional, List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch import nn, Tensor
import pdb
import time
from ..model_utils.transfusion_utils import clip_sigmoid
from ...utils import loss_utils
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned


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


class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        # pdb.set_trace()

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'heatmap' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class DETRHead(nn.Module):
    def __init__(self, model_cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu") -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.encoder = DSVT_Encoder(d_model, nhead, dim_feedforward, dropout,
                                        activation)

        # pdb.set_trace()
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.model_cfg.NUM_CLASSES, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(d_model, 64, 1, heads, use_bias=True)
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)


        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        # self.loss_heatmap = loss_utils.GaussianFocalLoss()
        # self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        self.loss_iou_rescore_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loss_iou_rescore_weight']
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'zhito')



        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.voxel_size = [ 0.32, 0.32, 10]
        self.grid_size = np.array([384, 264,   1])
        self.point_cloud_range = [-61.44, -42.24, -5.0, 61.44, 42.24, 5.0]
        if (model_cfg.get('DOWNSAMPLE_LAYER', None)):
            self.voxel_size = [self.voxel_size[0] * 2, self.voxel_size[1] * 2, self.voxel_size[2]]
            self.grid_size = np.array([round(self.grid_size[0] / 2), round(self.grid_size[1] / 2), self.grid_size[2]])
        self.num_classes = 6
        self.code_size = 8


        # self.downsampe_layer = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride = 2,padding=1, bias=False)
        # self.downsampe_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride = 2,padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )
        
        self.posembed=PositionEmbeddingLearned(2, 128)
        self.dirembed=PositionEmbeddingLearned(2, 128)

        self.query_embed = nn.Embedding(800, 128)
        # self.query_embed = None
        self.tgt = nn.Parameter(torch.rand(800, 128))

        self._reset_parameters()

    def _reset_parameters(self):
        # pdb.set_trace()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def dense2sequence(self, feautre):
        # pdb.set_trace()
        # feautre = self.downsampe_layer(feautre) #torch.Size([1, 128, 132, 192])
        # bs = feautre.size(0)
        # pdb.set_trace()

        x_grid, y_grid = feautre.shape[-2:]

        feautre_flatten = feautre.flatten(2,3)
        feature_abs = feautre_flatten.abs().sum(dim=1)
        feature_abs_flatten_idx = torch.nonzero(feature_abs > 0.)
        feautre_flatten_val = feautre_flatten.index_select(dim = 2, index=feature_abs_flatten_idx.permute(1,0)[1,...])

        top_proposals_x = feature_abs_flatten_idx.permute(1,0)[1,...] // y_grid # bs, num_proposals
        top_proposals_y = feature_abs_flatten_idx.permute(1,0)[1,...] % y_grid # bs, num_proposals

        top_proposals_x = (top_proposals_x - x_grid/2)
        top_proposals_y = (top_proposals_y - y_grid/2)
        dir_sin = torch.sin(torch.atan2(top_proposals_y,top_proposals_x))
        dir_cos = torch.cos(torch.atan2(top_proposals_y,top_proposals_x))
        dir_embed = self.dirembed(torch.cat([dir_sin.unsqueeze(1), dir_cos.unsqueeze(1)],dim=1).unsqueeze(0).float())
        pos_embed = self.posembed(torch.cat([top_proposals_x.unsqueeze(1), top_proposals_y.unsqueeze(1)],dim=1).unsqueeze(0).float())

        return feautre_flatten_val, pos_embed + dir_embed


    def forward(self, batch_dict):
        feature = batch_dict['spatial_features']
        feature = feature.permute(0,1,3,2).contiguous()
        bs = feature.size(0)

        res_layer = []
        for i in range(bs):
            feautre_flatten_val, feature_pos_embed = self.dense2sequence(feature[i,...].unsqueeze(0))
            # query_embed = None
            query_embed = self.query_embed.weight.unsqueeze(1)
            tgt = self.tgt.unsqueeze(1)
            # tgt = torch.zeros_like(query_embed)
            # pdb.set_trace()
            res_layer.append(self.encoder(tgt, feautre_flatten_val.permute(2,0,1), pos=feature_pos_embed.permute(2,0,1), query_pos=query_embed))

        res_layer = torch.cat(res_layer, dim=1)
        res_layer = res_layer.permute(1, 2, 0)
        #[1 128 800]
        res_layer = self.prediction_head(res_layer)
        
        # pdb.set_trace()
        if not self.training:
            bboxes = self.get_bboxes(res_layer)
            batch_dict['final_box_dicts'] = bboxes
            # pdb.set_trace()
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res_layer)


            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
            # pdb.set_trace()

        return batch_dict


    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap
        

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        # pdb.set_trace()
        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        # pdb.set_trace()
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)

        # pdb.set_trace()
        # visualization_feature(heatmap.squeeze(dim=0).cpu())
        # convert [bs,y,x] -> [bs,x,y] torch.Size([1, 6, 192, 132])
        heatmap = heatmap.permute(0,2,1)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None])

    def encode_bbox(self, bboxes):
        code_size = self.code_size
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2] + 0.5 * bboxes[:, 5]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        # pdb.set_trace()
        dim = dim.exp()
        height = height - dim[:, 2:3, :] * 0.5 
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        # pdb.set_trace()
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
                'cmask': cmask,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        # loss_heatmap = self.loss_heatmap(
        #     clip_sigmoid(pred_dicts["dense_heatmap"]),
        #     heatmap,
        # ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        # loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        # loss_all += loss_heatmap * self.loss_heatmap_weight
        # pdb.set_trace()
        # visualization_feature(heatmap[0].permute(0,2,1).squeeze(dim=0).cpu())

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER if head_name != 'iou'], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        # pdb.set_trace()
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        if "iou" in pred_dicts.keys():
            bbox_targets_for_iou = bbox_targets.permute(0, 2, 1)
            rot_iou = bbox_targets_for_iou[:, 6:8, :].clone()
            rot_iou = torch.atan2(rot_iou[:, 0:1, :], rot_iou[:, 1:2, :])
            dim_iou = bbox_targets_for_iou[:, 3:6, :].clone().exp()
            height_iou = bbox_targets_for_iou[:, 2:3, :].clone()
            center_iou = bbox_targets_for_iou[:, 0:2, :].clone()
            center_iou[:, 0, :] = center_iou[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            center_iou[:, 1, :] = center_iou[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            batch_box_targets_for_iou = torch.cat([center_iou, height_iou, dim_iou, rot_iou], dim=1).permute(0, 2, 1)

            rot_pred = pred_dicts['rot'].clone()
            center_pred = pred_dicts['center'].clone()
            height_pred = pred_dicts['height'].clone()
            rot_pred = torch.atan2(rot_pred[:, 0:1, :], rot_pred[:, 1:2, :])
            dim_pred = pred_dicts['dim'].clone().exp()
            center_pred[:, 0, :] = center_pred[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            center_pred[:, 1, :] = center_pred[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            batch_box_preds = torch.cat([center_pred, height_pred, dim_pred, rot_pred], dim=1).permute(0, 2, 1)

            batch_box_preds_for_iou = batch_box_preds.clone().detach()
            batch_box_targets_for_iou = batch_box_targets_for_iou.detach()
            layer_iou_loss = loss_utils.calculate_iou_loss_transfusionhead(
                iou_preds=pred_dicts['iou'],  
                batch_box_preds=batch_box_preds_for_iou,
                gt_boxes=batch_box_targets_for_iou,
                weights=bbox_weights,
                num_pos=num_pos
            )
            loss_all += (layer_iou_loss * self.loss_iou_rescore_weight)
            loss_dict[f"loss_iou"] = layer_iou_loss.item() * self.loss_iou_rescore_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all
        # pdb.set_trace()

        return loss_all,loss_dict


    def get_bboxes(self, preds_dicts):

        # pdb.set_trace()
        # feat_trt = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/inference_framework/combine_objs.npy').reshape(1,15,500)).cuda()
        # preds_dicts["center"] = feat_trt[:,0:2] #2
        # preds_dicts["height"] = feat_trt[:,2:3]#1
        # preds_dicts["dim"] = feat_trt[:,3:6]#3
        # preds_dicts["rot"] = feat_trt[:,6:8]#2
        # preds_dicts["iou"] = feat_trt[:,8:9]#1
        # preds_dicts["heatmap"] = feat_trt[:,9:15]#6


        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        # one_hot = F.one_hot(
        #     self.query_labels, num_classes=self.num_classes
        # ).permute(0, 2, 1)
        # pdb.set_trace()
        # batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
                                                                        # bare is a little lower
        # batch_score = batch_score  * preds_dicts["query_heatmap_score"] # batch_score from transformer via quert (sparse feature); but preds_dicts["query_heatmap_score"] and ont_hot from heatmap(dense feature)
        # tensor([[ 2],
        # [16],
        # [28]], device='cuda:0')

        # query_heatmap_score
        # tensor([[ 0],
        # [ 2],
        # [ 5],
        # [16],
        # [28],
        # [66]], device='cuda:0')

        # batch_score = batch_score
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        # pdb.set_trace()
        # visualization_feature(preds_dicts['dense_heatmap'][0].permute(0,2,1).sigmoid().squeeze(dim=0).cpu())
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]
        batch_iou = (preds_dicts['iou'] + 1) * 0.5 if 'iou' in preds_dicts else None
        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )

        if self.dataset_name == "nuScenes":
            self.tasks = [
                dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                dict(num_class=1, class_names=["pedestrian"], indices=[8], radius=0.175),
                dict(num_class=1,class_names=["traffic_cone"],indices=[9],radius=0.175),
            ]
        elif self.dataset_name == "Waymo":
            self.tasks = [
                dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                dict(num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7),
                dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
            ]
        elif self.dataset_name == "Zhito":
            self.tasks = [
                dict(num_class=3, class_names=[], indices=[0, 1, 2], radius=0.0175),
                dict(num_class=1, class_names=["pedestrian"], indices=[3], radius=0.0175),
                dict(num_class=1,class_names=["cyclist"],indices=[4],radius=0.0175),
                dict(num_class=1,class_names=["unknown"],indices=[5],radius=0.0175),
            ]

        new_ret_dict = []
        for i in range(batch_size):
            boxes3d = ret_dict[i]["pred_boxes"]
            scores = ret_dict[i]["pred_scores"]
            labels = ret_dict[i]["pred_labels"]
            cmask = ret_dict[i]['cmask']
            # IOU refine 
            if self.model_cfg.POST_PROCESSING.get('USE_IOU_TO_RECTIFY_SCORE', False) and batch_iou is not None:
                pred_iou = torch.clamp(batch_iou[i][0][cmask], min=0, max=1.0)
                IOU_RECTIFIER = scores.new_tensor(self.model_cfg.POST_PROCESSING.IOU_RECTIFIER)
                if len(IOU_RECTIFIER) == 1:
                    IOU_RECTIFIER = IOU_RECTIFIER.repeat(self.num_classes)
                scores = torch.pow(scores, 1 - IOU_RECTIFIER[labels]) * torch.pow(pred_iou, IOU_RECTIFIER[labels])
            
            keep_mask = torch.zeros_like(scores)
            for task in self.tasks:
                task_mask = torch.zeros_like(scores)
                for cls_idx in task["indices"]:
                    task_mask += labels == cls_idx
                task_mask = task_mask.bool()
                if task["radius"] > 0:
                    top_scores = scores[task_mask]
                    boxes_for_nms = boxes3d[task_mask][:, :7].clone().detach()
                    task_nms_config = copy.deepcopy(self.model_cfg.NMS_CONFIG)
                    task_nms_config.NMS_THRESH = task["radius"]
                    task_nms_config.NMS_PRE_MAXSIZE = task_nms_config.NMS_PRE_MAXSIZE[0]
                    task_nms_config.NMS_POST_MAXSIZE = task_nms_config.NMS_POST_MAXSIZE[0]
                    task_keep_indices, _ = model_nms_utils.class_agnostic_nms(
                            box_scores=top_scores, box_preds=boxes_for_nms,
                            nms_config=task_nms_config, score_thresh=task_nms_config.SCORE_THRES)
                else:
                    task_keep_indices = torch.arange(task_mask.sum())
                
                if task_keep_indices.shape[0] != 0:
                    keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                    keep_mask[keep_indices] = 1
            keep_mask = keep_mask.bool()
            ret = dict(pred_boxes=boxes3d[keep_mask], pred_scores=scores[keep_mask], pred_labels=labels[keep_mask])
            new_ret_dict.append(ret)

        for k in range(batch_size):
            new_ret_dict[k]['pred_labels'] = new_ret_dict[k]['pred_labels'].int() + 1

        return new_ret_dict 



class DSVT_Encoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.num_layers = 3
        self.return_intermediate = False
    
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = _get_clones(decoder_layer, self.num_layers)
        


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        # pdb.set_trace()

        for layer in self.layers:
            # pdb.set_trace()
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # pdb.set_trace()
        # query = torch.zeros([1,100,128], device=q.device, requires_grad=True)
        # tgt2 = self.multihead_attn(query=query,key=memory.flatten(0,1).unsqueeze(0),value=memory.flatten(0,1).unsqueeze(0), attn_mask=memory_mask,key_padding_mask=memory_key_padding_mask.flatten(0).unsqueeze(0))
        # torch.Size([1, 100, 128])

        # memory = F.max_pool1d(memory.permute(2,1,0), kernel_size=2, stride=2, padding=0).permute(2,1,0)
        # memory_key_padding_mask = F.max_pool1d(memory_key_padding_mask.float(), kernel_size=2, stride=2, padding=0).bool()

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
