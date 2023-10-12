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
# import sys
# sys.path.insert(0, '/data/zhenghu/local/DSVT')
# sys.path.remove('/data/zhenghu/local/OpenPCDet')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        data_file_list, label_file_list = self.init_data_list(str(self.root_path/ 'car_pedestrian_more_new' /'j6_test_202300728_car_30000_pedestrian_15000-5000-1000-6000.txt'))
        self.sample_file_list = data_file_list
        self.label_file_list = label_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # index = 39
            # index = 36
            # index = 83
            data_path = str(self.root_path) + self.sample_file_list[index]
            # data_path = '/home/zhenghu/datasets/zhito/J6/choosed_lidar_data_20211016_j6a4_urban_rainy/mindflow_14_2520_1/sequence069/pcl/bin/1634360203.995325.bin'
            # data_path = '/home/zhenghu/datasets/zhito/J6/choosed_lidar_data_20220318pm_j6a1_urban_shujijinglu_augment_ros_2hz/mindflow_24_2738/sequence035/pcl/bin/1647592909.218359.bin'
            # print('path: ', data_path)
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


    def output_result(self, result, index):
        
        ## init path and read label file
        output_root = "./results/Results_Dsvt_ZhitoData_AllCls_J6_0816_02_5W_150epoch_lr002_range_adam_threshold02"
        label_file = (str(self.root_path) + self.label_file_list[index]).replace('\n','')
        result_file = Path(os.getcwd()).parent / (output_root + self.label_file_list[index]).replace('\n','')
        with open(label_file, 'r', encoding='utf-8') as j:
            contents = json.load(j)

        if len(contents['labels']) > 0:
            label_template =  contents['labels'][0].copy()
        else:
            label_template = result_template.copy()

        ## transfer output to label format
        class_names = self.class_names
        score_list = result['pred_scores'].detach().cpu().numpy().tolist()
        bbox_list = result['pred_boxes'].detach().cpu().numpy().tolist()
        class_list = result['pred_labels'].detach().cpu().numpy().tolist()

        num_obj = len(score_list)
        labels = []
        for i in range(num_obj):

            label = label_template.copy()
            label['ID'] = i
            label['category'] = class_names[class_list[i] - 1] # classfier start at 1
            label['track_id'] = ''
            label['location'] = [bbox_list[i][0], bbox_list[i][1], bbox_list[i][2]]
            label['size'] = [bbox_list[i][3], bbox_list[i][4], bbox_list[i][5]]
            label['rotation'] = [0, 0, bbox_list[i][6]]
            label['num_points'] = 0
            label['occlusion'] = ''
            label['lidar']=[]
            label['sub_category'] = ''
            label['union_ID']=''
            label['score'] = score_list[i]
            # print(score_list[i], score_list[i].dtype)
            labels.append(label)

        contents['labels'] = labels
        
        ## output to result file
        if not os.path.exists(result_file.parent):
            os.makedirs(result_file.parent)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(contents, f)

        # print(index, '---', result_file)



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

    # pdb.set_trace()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            # if idx < 83:
            #     continue
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # pdb.set_trace()

            demo_dataset.output_result(pred_dicts[0], idx)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], gt_boxes=demo_dataset.gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
