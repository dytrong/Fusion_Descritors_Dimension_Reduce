#! /bin/zsh
dataset='i'
dimension=256
###1152, 2176, 4224
fusion_dimension=4224
EPOCH=500
pre_trained_cnn='densenet169'
###cnn, sigmoid, tanh
autoencoder_type='sigmoid'
batch_size=32
lr=0.0001

####融合描述幅的位置
pre_trained_descs_path="./fusion_descriptors/${pre_trained_cnn}/${dataset}_${fusion_dimension}_${pre_trained_cnn}_hardnet_all_descs.npy"

echo "融合描述符的位置: ${pre_trained_descs_path}"

while [ ${dimension} != '512' ]
do

###每隔100 epoch保存一次模型参数
autoencoder_parameter_path="${fusion_dimension}_train_test_${dataset}_${pre_trained_cnn}_hardnet_${autoencoder_type}_${dimension}_${batch_size}_${lr}.pth"

echo "模型参数保存的位置:${autoencoder_parameter_path}"

(CUDA_VISIBLE_DEVICES=1  python -u  train_autoencoder_fusion.py \
--dataset ${dataset} \
--pre_trained_cnn ${pre_trained_cnn} \
--dimension ${dimension} \
--pre_trained_descs_path ${pre_trained_descs_path} \
--autoencoder_type ${autoencoder_type} \
--autoencoder_parameter_path ${autoencoder_parameter_path} \
--EPOCH ${EPOCH} \
--batch_size ${batch_size} \
--lr ${lr} \
> "./log/${EPOCH}_${fusion_dimension}_train_test_${dataset}_${pre_trained_cnn}_hardnet_${autoencoder_type}_${dimension}_${batch_size}_${lr}.txt"
)
dimension=$((dimension*2))
done
