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
import json
import pdb
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import struct
import torch.nn as nn

result_template = {
          "ID": 0, 
          "category": "car", 
          "track_id": 0, 
          "location": [0.0, 0.0, 0.0], 
          "size": [0.0, 0.0, 0.0], 
          "rotation": [0.0, 0.0, 0.0],
          "num_points": [0],
          "occlusion": 0,
          "lidar": [0],
          "sub_category": "",
          "union_ID": "",
          "score":0.0
}


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
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # data_file_list.sort()

        # pdb.set_trace()
        data_file_list, label_file_list = self.init_data_list(str(self.root_path /'j6_test_20220525_10000.txt'))
        self.sample_file_list = data_file_list
        self.label_file_list = label_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # index = 39
            # index = 36
            # index = 83
            # index = 91
            data_path = str(self.root_path) + self.sample_file_list[index]
            # data_path = '/home/zhenghu/datasets/zhito/J6/choosed_lidar_data_20211016_j6a4_urban_rainy/mindflow_14_2520_1/sequence069/pcl/bin/1634360203.995325.bin'
            # data_path = '/home/zhenghu/datasets/zhito/J6/choosed_lidar_data_20220318pm_j6a1_urban_shujijinglu_augment_ros_2hz/mindflow_24_2738/sequence035/pcl/bin/1647592909.218359.bin'
            print('path: ', data_path)
            label_path = str(self.root_path) + self.label_file_list[index].replace('\n','')
            self.gt_boxes = self.init_gtboxes_data(label_path)

            # pdb.set_trace()
            points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)
            points[:,-1] = points[:,-1]/255.
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        # pdb.set_trace()
        return data_dict


    def init_gtboxes_data(self, path):
        with open(path, 'r', encoding='utf-8') as j:
            contents = json.load(j)
        if len(contents['labels']) > 0:
            eles = []
            for i in range(len(contents['labels'])):
                ele = contents['labels'][i]['location'] + contents['labels'][i]['size'] + [contents['labels'][i]['rotation'][-1]]
                # pdb.set_trace()
                if (max(abs(np.array(ele))) > 74.88):
                    continue
                eles.append(ele)
            return np.array(eles)
    
    def init_data_list(self, data_dir):
        label_list = []
        if data_dir.split('.')[-1] == 'txt': 
            with open(data_dir, "r") as f:
                lines = f.readlines()
            data_list = []
            for i, line in enumerate(lines):
                pcd_path, label_path = line.split('\t')
                data_list.append(pcd_path)
                label_list.append(label_path)
            data_dir = str(Path(data_dir).parent)
            # data_dir = str(Path(data_dir).parent)
            # print(self.data_dir)
        else:
            data_list = [filename for filename in os.listdir(data_dir)]
            data_list.sort()
        return data_list, label_list

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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    model.cuda()
    model.eval()
    pdb.set_trace()
    with open('dsvt_zhito_2dsvt_1_2rpn.wts', 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print('**'*10)
            print(k)
            # print(v)
            print(v.shape)
            # pdb.set_trace()
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')


if __name__ == '__main__':
    main()
