# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=combine3modules_dynamic_shape.onnx \
# --saveEngine=combine3modules_dynamic_shape.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x128,voxel_coords:3000x4 --optShapes=voxel_features:25000x128,voxel_coords:25000x4 --maxShapes=voxel_features:35000x128,voxel_coords:35000x4


# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=combine3modules_dynamic_shape.onnx \
# --saveEngine=combine3modules_dynamic_shape_fp16.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x128,voxel_coords:3000x4 \
# --optShapes=voxel_features:25000x128,voxel_coords:25000x4 \
# --maxShapes=voxel_features:30000x128,voxel_coords:30000x4

/home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=combine3modules_dynamic_shape_noscatter_zhito.onnx \
--saveEngine=combine3modules_dynamic_shape_noscatter_fp16_zhito.engine \
--memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 \
--tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS 

# Unique operator is not support in TensorRT,need to implement oneself.
# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=dsvt_blocks.onnx \
# --saveEngine=dsvt_blocks.engine \
# --memPoolSize=workspace:4096 --buildOnly --device=0 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=src:1000x128,set_voxel_inds_tensor_shift_0:2x50x36,set_voxel_inds_tensor_shift_1:2x50x36,set_voxel_masks_tensor_shift_0:2x50x36,set_voxel_masks_tensor_shift_1:2x50x36,pos_embed_tensor:4x2x1000x128 \
# --optShapes=src:24629x128,set_voxel_inds_tensor_shift_0:2x1156x36,set_voxel_inds_tensor_shift_1:2x834x36,set_voxel_masks_tensor_shift_0:2x1156x36,set_voxel_masks_tensor_shift_1:2x834x36,pos_embed_tensor:4x2x24629x128 \
# --maxShapes=src:100000x128,set_voxel_inds_tensor_shift_0:2x5000x36,set_voxel_inds_tensor_shift_1:2x3200x36,set_voxel_masks_tensor_shift_0:2x5000x36,set_voxel_masks_tensor_shift_1:2x3200x36,pos_embed_tensor:4x2x100000x128


/home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=./dsvt_blocks_zhito.onnx \
--saveEngine=./dsvt_blocks_zhito.engine \
--memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 \
--tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
--minShapes=src:2000x128,set_voxel_inds_tensor_shift_0:2x100x36,set_voxel_inds_tensor_shift_1:2x80x36,set_voxel_masks_tensor_shift_0:2x100x36,set_voxel_masks_tensor_shift_1:2x80x36,pos_embed_tensor:4x2x2000x128 \
--optShapes=src:12000x128,set_voxel_inds_tensor_shift_0:2x600x36,set_voxel_inds_tensor_shift_1:2x400x36,set_voxel_masks_tensor_shift_0:2x600x36,set_voxel_masks_tensor_shift_1:2x400x36,pos_embed_tensor:4x2x12000x128 \
--maxShapes=src:25000x128,set_voxel_inds_tensor_shift_0:2x1200x36,set_voxel_inds_tensor_shift_1:2x1000x36,set_voxel_masks_tensor_shift_0:2x1200x36,set_voxel_masks_tensor_shift_1:2x1000x36,pos_embed_tensor:4x2x25000x128
# > debug.log 2>&1


# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=./pos_enbedding_zhito_dynamic.onnx \
# --saveEngine=./pos_enbedding_zhito_dynamic.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=coors_in_win_shift0:2000x3,coors_in_win_shift1:2000x3 \
# --optShapes=coors_in_win_shift0:12000x3,coors_in_win_shift1:12000x3 \
# --maxShapes=coors_in_win_shift0:25000x3,coors_in_win_shift1:25000x3,

/home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=./pos_embedding_zhito.onnx \
--saveEngine=./pos_embedding_zhito.engine \
--memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 \
--tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS
# > debug.log 2>&1


# for self-attention
# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=selfattention.onnx \
# --saveEngine=selfattention.engine \
# --memPoolSize=workspace:4096 --buildOnly --device=0 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=query:100x36x128,key:100x36x128,value:100x36x128,key_padding_mask:100x36 \
# --optShapes=query:1156x36x128,key:1156x36x128,value:1156x36x128,key_padding_mask:1156x36 \
# --maxShapes=query:2000x36x128,key:2000x36x128,value:2000x36x128,key_padding_mask:2000x36



#  &>result.txt