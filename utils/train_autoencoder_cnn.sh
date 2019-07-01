#! /bin/zsh
dataset='v'
dimension=128
model_type='resnet101'
EPOCH=50

####训练数据集地址
dataset_path="/home/data/daizhuang/dataset/all_desc/${model_type}/${dataset}_${model_type}_all_descs.npy"

while [ ${dimension} != '8192' ]
do
###Auto-Encoder模型参数地址
model_parameters_path="./parameters/model_parameters/${model_type}/${EPOCH}_epoch/all_descs/${EPOCH}_${dataset}_${model_type}_${dimension}_autoencoder_cnn.pth"
(CUDA_VISIBLE_DEVICES=0  python -u  train_autoencoder_cnn.py \
--dataset ${dataset} \
--dimension ${dimension} \
--model_type ${model_type} \
--EPOCH ${EPOCH} \
--dataset_path ${dataset_path} \
--model_parameters_path ${model_parameters_path} > ./log/${model_type}/${EPOCH}_epoch/${EPOCH}_${dataset}_${model_type}_${dimension}.txt
)
dimension=$((dimension*2))
done
