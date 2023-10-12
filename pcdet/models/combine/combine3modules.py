# by zhenghu

import torch
import torch.nn as nn
# from models.backbones_2d.map_to_bev.pointpillar3d_scatter import PointPillarScatter3d
# from models.backbones_2d.base_bev_res_backbone import BaseBEVResBackbone
from ..dense_heads import CenterHead
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper
import pdb
import numpy as np


class Combine3Modules(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        # self.pillarscatter = PointPillarScatter3d
        # self.backbone2d = BaseBEVResBackbone
        self.center_head = CenterHead
        # self.shared_conv = self.center_head.shared_conv
        # self.separate_heads = self.center_head.heads_list[0]



        input_names = [
            'input',
        ]
        output_names = ['center', 
                        'center_z', 
                        'dim', 
                        'rot', 
                        'iou', 
                        'hm',
        ]
        trt_path = '/home/zhenghu/DeepLearning/DSVT/tools/combine3modules_dynamic_shape_noscatter_fp16_zhito.engine'
        # trt_path = '/home/zhenghu/DeepLearning/DSVT/tools/combine3modules_dynamic_shape_noscatter_fp16_waymo.engine'
        self.allcombinetrt = TRTWrapper(trt_path, input_names, output_names)


    def forward(self, batch_dict):
        # spatial_features = self.pillarscatter(pillar_features, coords)
        # spatial_features_2d = self.backbone2d(spatial_features)
        # feats = self.shared_conv(spatial_features_2d)
        # dense_preds = self.separate_heads(feats)

        # pillar_features = batch_dict['pillar_features']
        # coords = batch_dict['voxel_coords']

        # inputs_dict = dict(
        #         voxel_features=pillar_features,
        #         voxel_coords=coords,
        #     )
        # pdb.set_trace()
        output = self.allcombinetrt({'input':batch_dict['pillar_features'].permute(0,2,3,1)})

        # output = dict(
        #         center = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/center.npy').reshape(1, 2, 468, 468)).cuda(),
        #         center_z = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/center_z.npy').reshape(1, 1, 468, 468)).cuda(),
        #         dim = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/dim.npy').reshape(1, 3, 468, 468)).cuda(),
        #         rot = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/rot.npy').reshape(1, 2, 468, 468)).cuda(),
        #         iou = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/iou.npy').reshape(1, 1, 468, 468)).cuda(),
        #         hm = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/combine_output_layer/hm.npy').reshape(1, 6, 468, 468)).cuda(),
        #     )
        # pdb.set_trace()


        
        # for item in output:
        #     batch_dict[item] = output[item]

        batch_dict['output'] = output
        return batch_dict
