# python demo.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml \
# --ckpt /home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth \
# --data_path /home/zhenghu/datasets/waymo/waymo_format/waymo/waymo_processed_data_v0_5_0/segment-1022527355599519580_4866_960_4886_960_with_camera_labels

python demo.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_trtengine.yaml \
--ckpt /home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth \
--data_path /home/zhenghu/datasets/waymo/waymo_format/waymo/test
