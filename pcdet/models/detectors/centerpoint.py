from .detector3d_template import Detector3DTemplate
import pdb
import time
import torch


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # pdb.set_trace()
        t0 = time.time()

        # batch_dict = self.module_list[0](batch_dict)
        # # pdb.set_trace()
        # batch_dict = self.module_list[1](batch_dict)
        # batch_dict = self.module_list[2](batch_dict)
        # batch_dict = self.module_list[3](batch_dict)
        # batch_dict = self.module_list[4](batch_dict)
        for cur_module in self.module_list:
            # pdb.set_trace()
            batch_dict = cur_module(batch_dict)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     # print(cur_module)
        #     # print('cost time: ', t1-t0)
        #     t0 = time.time()

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            # loss, tb_dict, disp_dict = self.get_training_transhead_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            # pdb.set_trace()
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            t1 = time.time()
            # print('post_process: ', t1-t0)
            return pred_dicts, recall_dicts

    def get_training_transhead_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict


    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            # 'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        # pdb.set_trace()
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
