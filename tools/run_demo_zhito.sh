# python demo_zhito.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito.yaml \
# --ckpt /home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito/default/ckpt/latest_model.pth \
# --data_path /home/zhenghu/datasets/zhito/J6 \
# --ext .bin

# python demo_zhito.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_trtengine.yaml \
# --ckpt /home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito/default/ckpt/latest_model.pth \
# --data_path /home/zhenghu/datasets/zhito/J6 \
# --ext .bin



python demo_zhito.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range.yaml \
--ckpt /home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito/default/ckpt/latest_model_4dsvt.pth \
--data_path /home/zhenghu/datasets/zhito/J6 \
--ext .bin


# python demo_zhito.py --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_zhito_5w_150epoch_lr002_range_adam_300epoch_2dsvt.yaml \
# --ckpt /home/zhenghu/DeepLearning/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_zhito/default/ckpt/latest_model_2dsvt.pth \
# --data_path /home/zhenghu/datasets/zhito/J6 \
# --ext .bin
