from .detector3d_template import Detector3DTemplate
import pdb
import time
import torch
import numpy as np


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # pdb.set_trace()
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # pdb.set_trace()
        t0 = time.time()

        DynamicPillarVFE_3d = self.module_list[0]
        DSVT = self.module_list[1]
        PointPillarScatter3d = self.module_list[2]
        BaseBEVResBackbone = self.module_list[3]
        CenterHead = self.module_list[4]
        # pdb.set_trace()

        batch_dict = DynamicPillarVFE_3d(batch_dict)
        torch.cuda.synchronize()
        t1 = time.time()

        batch_dict = DSVT(batch_dict)
        torch.cuda.synchronize()
        t2 = time.time()
        # pdb.set_trace()
        if 6 == len(self.module_list):
            batch_dict['pillar_features'] = self.module_list[3](batch_dict['voxel_features'], batch_dict['voxel_coords'])
            # pdb.set_trace()
            # batch_dict['pillar_features'] = torch.from_numpy(np.load('/home/zhenghu/DeepLearning/DSVT/tools/npy_file/scatter_output_layer/scatter.npy').reshape(1, 468, 468, 128)).cuda()
            # batch_dict['pillar_features'] = batch_dict['pillar_features'].permute(0,3,1,2)
            
            batch_dict = self.module_list[2](batch_dict)
            # batch_dict['spatial_features'] = batch_dict['pillar_features']
            # pdb.set_trace()
            torch.cuda.synchronize()
            t3 = time.time()
            self.CenterHead = self.module_list[5]
            torch.cuda.synchronize()
            t4 = time.time()
            t5 = time.time()
        else:
        # batch_dict['pillar_features'] = batch_dict['voxel_features']
            # batch_dict = PointPillarScatter3d(batch_dict)
            batch_dict['pillar_features'] = PointPillarScatter3d(batch_dict['voxel_features'], batch_dict['voxel_coords'])
            batch_dict['spatial_features'] = batch_dict['pillar_features']
            torch.cuda.synchronize()
            t3 = time.time()
            # batch_dict = BaseBEVResBackbone(batch_dict)
            batch_dict['spatial_features_2d'] = BaseBEVResBackbone(batch_dict['spatial_features'])
            torch.cuda.synchronize()
            t4 = time.time()
            batch_dict = CenterHead(batch_dict)
            torch.cuda.synchronize()
            t5 = time.time()
        # print('cost time: ', t1-t0)
        # print('cost time: ', t2-t1)
        # print('cost time: ', t3-t2)
        # print('cost time: ', t4-t3)
        # print('cost time: ', t5-t4)



        # for cur_module in self.module_list:
        #     # pdb.set_trace()
        #     if 'DSVT' == type(cur_module).__name__:
        #         batch_dict['voxel_features'] = cur_module(batch_dict['voxel_features'], batch_dict['voxel_coords'])
        #         batch_dict['pillar_features'] = batch_dict['voxel_features']
        #         continue
        #     batch_dict = cur_module(batch_dict)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     # print(cur_module)
        #     print('cost time: ', t1-t0, type(cur_module).__name__)
        #     t0 = time.time()
# cost time:  0.03305554389953613
# cost time:  0.14070343971252441
# cost time:  0.006891965866088867
# cost time:  0.11799454689025879
# cost time:  0.05345869064331055

# cost time:  0.021335601806640625
# cost time:  0.14223885536193848
# cost time:  0.00414276123046875
# cost time:  0.11783051490783691
# cost time:  0.05111408233642578

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # pred_dicts, recall_dicts = self.post_processing_trt(batch_dict)
            t1 = time.time()
            print('post_process: ', t1-t0)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict



    def post_processing_trt(self, batch_dict):
        pred_dicts = []
        pred_dicts.append(batch_dict['output'])
        pred_dicts = self.CenterHead.generate_predicted_boxes(batch_dict['batch_size'], pred_dicts)
        batch_dict['final_box_dicts'] = pred_dicts

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
        return final_pred_dict, recall_dict
