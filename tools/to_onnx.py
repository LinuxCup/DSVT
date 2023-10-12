import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pdb

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        # pdb.set_trace()

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    # pdb.set_trace()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    weight = torch.load(args.ckpt)
    model.load_state_dict(weight['model_state'])
    model.cuda()
    model.eval()

    dummy_input = torch.randn(1, 8, 320, 640).cuda()

    input1 = torch.randn(20254, 128).cuda()
    input2 = torch.LongTensor(962, 36).to(torch.device('cuda'))
    # input2 = torch.LongTensor(962, 36)
    input2.fill_(0)
    input3 = torch.randn(962, 36).cuda()
    input4 = torch.randn(20254, 128).cuda()
    input_name1 = "input_name1"
    input_name2 = "input_name2"
    input_name3 = "input_name3"
    input_name4 = "input_name4"
    output_names = "output_name1"
    encoder=model.backbone_3d.stage_0[0].encoder_list[0]
    input = (input1, input2, input3, input4)
    # pdb.set_trace()
    # input = torch.randn(1, 128, 468, 468).cuda()
    # torch.onnx.export(model.backbone_2d, input, "original.onnx", verbose=False, input_names=['input1'], output_names=['output1'])
    torch.onnx.export(encoder, input, "dsvt_layer.onnx", 
                      opset_version = 11,
                      input_names=[input_name1, input_name2, input_name3, input_name4], 
                      output_names=[output_names],
                    #   dynamic_axes = {
                    #     input_name1: {0: 'sequence_length'},
                    #     input_name2: {0: 'batch_size'},
                    #     input_name3: {0: 'batch_size'},
                    #     input_name4: {0: 'sequence_length'},
                    #     output_names: {0: 'sequence_length'}}
                      )
    
    
    pdb.set_trace()
    block = model.backbone_3d.stage_0
    input1 = torch.randn(20254, 128).cuda()
    input2 = [torch.LongTensor(2, 962, 36).to(torch.device('cuda')), torch.LongTensor(2, 962, 36).to(torch.device('cuda'))]
    input2[0].fill_(0)
    input2[1].fill_(0)
    input3 = [torch.BoolTensor(2, 962, 36).to(torch.device('cuda')), torch.BoolTensor(2, 962, 36).to(torch.device('cuda'))]
    input4 = [torch.randn(20254, 128).cuda(), torch.randn(20254, 128).cuda()]
    input_name5 = 'input_name5'
    input = (input1, input2, input3, input4, 0)
    torch.onnx.export(block, input, "dsvt_block.onnx", 
                      opset_version = 11,
                      input_names=[input_name1, input_name2, input_name3, input_name4, input_name5], 
                      output_names=[output_names],
                      )
    
    
    pdb.set_trace()
    # dsvt = model.backbone_3d
    # input1 = torch.randn(20254, 128).cuda()
    # input2 = torch.LongTensor(20254, 4).to(torch.device('cuda'))
    # input2.fill_(0)
    # input = (input1, input2)
    # pdb.set_trace()
    # torch.onnx.export(block, input, "dsvt.onnx", 
    #                   opset_version = 11,
    #                   input_names=[input_name1, input_name2], 
    #                   output_names=[output_names],
    #                   )
    # pdb.set_trace()
    

    # model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pdb.set_trace()

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
