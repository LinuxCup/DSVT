# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=combine3modules_dynamic_shape.onnx \
# --saveEngine=combine3modules_dynamic_shape.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x128,voxel_coords:3000x4 --optShapes=voxel_features:25000x128,voxel_coords:25000x4 --maxShapes=voxel_features:35000x128,voxel_coords:35000x4


# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=combine3modules_dynamic_shape.onnx \
# --saveEngine=combine3modules_dynamic_shape_fp16.engine \
# --memPoolSize=workspace:4096 --verbose --buildOnly --device=0 --fp16 --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=voxel_features:3000x128,voxel_coords:3000x4 --optShapes=voxel_features:25000x128,voxel_coords:25000x4 --maxShapes=voxel_features:35000x128,voxel_coords:35000x4


# Unique operator is not support in TensorRT,need to implement oneself.
/home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=dsvt_blocks.onnx \
--saveEngine=dsvt_blocks.engine \
--memPoolSize=workspace:4096 --buildOnly --device=0 --fp16 \
--tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
--minShapes=src:1000x128,set_voxel_inds_tensor_shift_0:2x50x36,set_voxel_inds_tensor_shift_1:2x50x36,set_voxel_masks_tensor_shift_0:2x50x36,set_voxel_masks_tensor_shift_1:2x50x36,pos_embed_tensor:4x2x1000x128 \
--optShapes=src:24629x128,set_voxel_inds_tensor_shift_0:2x1156x36,set_voxel_inds_tensor_shift_1:2x834x36,set_voxel_masks_tensor_shift_0:2x1156x36,set_voxel_masks_tensor_shift_1:2x834x36,pos_embed_tensor:4x2x24629x128 \
--maxShapes=src:100000x128,set_voxel_inds_tensor_shift_0:2x5000x36,set_voxel_inds_tensor_shift_1:2x3200x36,set_voxel_masks_tensor_shift_0:2x5000x36,set_voxel_masks_tensor_shift_1:2x3200x36,pos_embed_tensor:4x2x100000x128


# /home/zhenghu/Downloads/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/bin/trtexec --onnx=dsvt_blocks.onnx \
# --saveEngine=dsvt_blocks.engine \
# --memPoolSize=workspace:40960 --verbose --buildOnly --device=0 --fp16 \
# --tacticSources=+CUDNN,+CUBLAS,-CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS \
# --minShapes=src:1000x128,set_voxel_inds_tensor_shift_0:2x1000x36,set_voxel_inds_tensor_shift_1:2x1000x36,set_voxel_masks_tensor_shift_0:2x1000x36,set_voxel_masks_tensor_shift_1:2x1000x36,pos_embed_tensor:4x2x1000x128 \
# --optShapes=src:30000x128,set_voxel_inds_tensor_shift_0:2x30000x36,set_voxel_inds_tensor_shift_1:2x30000x36,set_voxel_masks_tensor_shift_0:2x30000x36,set_voxel_masks_tensor_shift_1:2x30000x36,pos_embed_tensor:4x2x30000x128 \
# --maxShapes=src:100000x128,set_voxel_inds_tensor_shift_0:2x100000x36,set_voxel_inds_tensor_shift_1:2x100000x36,set_voxel_masks_tensor_shift_0:2x100000x36,set_voxel_masks_tensor_shift_1:2x100000x36,pos_embed_tensor:4x2x100000x128

#  &>result.txt