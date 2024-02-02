python gen_wts_test.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn_sparse_pre.yaml \
--ckpt /home/zhenghu/DeepLearning/DSVT_train/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt_1_2rpn_sparse_pre/default/ckpt/latest_model.pth \
--data_path /home/zhenghu/datasets/zhito/J6 \
--ext .bin


# python gen_wts_test.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml \
# --ckpt /home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth \
# --data_path /home/zhenghu/datasets/zhito/J6 \
# --ext .bin

# python demo.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_trtengine.yaml --ckpt /home/zhenghu/DeepLearning/DSVT/output/cfgs/dsvt_models/dsvt_plain_1f_onestage/default/ckpt/latest_model.pth --data_path /home/zhenghu/datasets/waymo/waymo_format/waymo/test
