import pdb
import struct
import sys
import torch


import argparse
import logging
import os
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
# from det3d import torchie
# from det3d.core import coco_eval, results2json
# from det3d.datasets import  build_dataset
# from det3d.datasets.kitti import kitti_common as kitti
# from det3d.datasets.kitti.eval import get_official_eval_result
# from det3d.datasets.utils.kitti_object_eval_python.evaluate import (evaluate as kitti_evaluate,)
# from det3d.models import build_detector
# from det3d.torchie.apis import init_dist
# from det3d.torchie.apis.train import example_convert_to_torch
# from det3d.torchie.parallel import MegDataParallel, MegDistributedDataParallel
# from det3d.torchie.trainer import get_dist_info, load_checkpoint
# from det3d.torchie.trainer.trainer import example_to_device
# from det3d.utils.dist.dist_common import (all_gather, get_rank, get_world_size, is_main_process, synchronize,)
# from tqdm import tqdm
# from det3d.torchie.parallel import collate, collate_kitti
# from torch.utils.data import DataLoader
import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("--config", default='../examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py', help="test config file path")
    parser.add_argument("--checkpoint", default='cia-ssd-model.pth',  help="checkpoint file")
    parser.add_argument("--out", default='out.pkl', help="output result file")
    parser.add_argument("--json_out",  default='json_out.json', help="output result file name without extension", type=str)
    parser.add_argument("--eval", type=str, nargs="+", choices=["proposal", "proposal_fast", "bbox", "segm", "keypoints"], help="eval types",)
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--txt_result", default=True, help="save txt")
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none",help="job launcher",)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    

    parser.add_argument('--fp16', action='store_true', default=False, help='trigger mixed precision')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    pdb.set_trace()

    args = parse_args()
    print(args)
    assert args.out or args.show or args.json_out, ('Please specify at least one operation (save or show the results) with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    if args.json_out is not None and args.json_out.endswith(".json"):
        args.json_out = args.json_out[:-5]

    cfg = torchie.Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark", False):  # False
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint_path = os.path.join(cfg.work_dir, args.checkpoint)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to('cuda:0')
    model.eval()

    with open('dsvt.wts', 'w') as f:
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

if __name__ == "__main__":
    main()




