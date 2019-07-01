#! /bin/zsh
dataset='v'
dimension=4096
reduce_method='AUTO'
EPOCH=50
model_name='densenet169'
fusion_method='cat'


####-u是为了禁止缓存，让结果可以直接进入日志文件
while [ ${dimension} != '8192' ]
do
######auto encoder 预训练模型地址
autoencoder_parameter_path="../../utils/parameters/model_parameters/${model_name}/${EPOCH}_epoch/all_descs/${dataset}/${EPOCH}_${dataset}_${model_name}_${dimension}_autoencoder_cnn.pth"
#####训练数据集地址
desc_path="../../utils/parameters/descriptors/${model_name}/${dataset}_${model_name}_all_descs.npy"

(CUDA_VISIBLE_DEVICES=1  python -u  generate_fusion_descs.py \
--dataset ${dataset} \
--dimension ${dimension} \
--reduce_method ${reduce_method} \
--autoencoder_parameter_path ${autoencoder_parameter_path} \
--pre_trained_model ${model_name} \
--pre_trained_descs_path ${desc_path} \
--fusion_method ${fusion_method} \
> "./log/${dataset}_${model_name}_fusion_${dimension}_log.txt"
)
dimension=$((dimension*2))
done
