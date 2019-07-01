import torch 
from torch import nn 
import torchvision 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np
import time
import argparse
from autoencoder_models import *

###reproducible
torch.manual_seed(1)

def normalization(input_data):
    mean_data = input_data.mean(axis=1, keepdims=True)
    ####ddof是自由度, 等于１是算的是样本方差
    ####默认是等于0,算的是总体方差
    std_data = input_data.std(axis=1, keepdims=True)
    print('样本均值为:'+str(mean_data))
    print('样本方差为:'+str(std_data))
    output_data = (input_data - mean_data) / std_data
    return output_data

####将数据归一化到[0,1]之间
def regularization(input_data):
    MAX_NUM = np.max(input_data)
    MIN_NUM = np.min(input_data)
    print('样本最大值为:'+str(MAX_NUM))
    print('样本最小值为:'+str(MIN_NUM))
    output_data = (input_data - MIN_NUM) / float(MAX_NUM - MIN_NUM)
    ###归一化到[-1,1]
    output_data = (output_data - 0.5) / 0.5
    return output_data

def random_select_samples(train_data, sample_number):
    row_rand_array = np.arange(train_data.shape[0])
    np.random.shuffle(row_rand_array)
    train_data = train_data[row_rand_array[0:sample_number]]
    train_data = train_data.astype('float32')
    return train_data

def change_data_to_fit_model(train_data, dataset_type):
    #####将数据转化为tensor
    train_data = torch.from_numpy(train_data).float()
    ######alexnet数据集
    if dataset_type == 'alexnet':
        train_data = train_data.view(train_data.size(0), 256, 11, 11)
    ######vgg16数据集
    if dataset_type == 'vgg16':
        train_data = train_data.view(train_data.size(0), 512, 26, 26)
    ######resnet101数据集
    if dataset_type == 'resnet101':
        train_data = train_data.view(train_data.size(0), 1024, 12, 12)
    ######Densenet169数据集
    if dataset_type == 'densenet169':
        train_data = train_data.view(train_data.size(0), 1280, 12, 12)
    print('训练autoencoder数据集大小为:'+str(train_data.shape))
    return train_data

###下载训练数据集,对所有特征点进行训练
def load_trained_data(dataset_path,dataset_type):
    start = time.time()
    train_data = np.load(dataset_path)
    print("下载数据集共耗时:"+str(time.time()-start))
    ####随机抽取数据集多少行,因为vgg16描述符太大, 占用内存超过电脑内存
    if dataset_type == 'vgg16':
        train_data = random_select_samples(train_data, 9000)
    start = time.time()
    train_data = normalization(train_data)
    print("标准化数据集共耗时:"+str(time.time()-start))
    ######将.npy转化为适应模型的格式
    train_data = change_data_to_fit_model(train_data, dataset_type)
    return train_data

def select_pretrained_model(model_type, dimension):
    if model_type == 'alexnet':
        AutoEncoder = AutoEncoder_alexnet(dimension).cuda()
    if model_type == 'vgg16':
        AutoEncoder = AutoEncoder_vgg16(dimension).cuda()
    if model_type == 'resnet101':
        AutoEncoder = AutoEncoder_resnet101(dimension).cuda()
    if model_type == 'densenet169':
        AutoEncoder = AutoEncoder_densenet169(dimension).cuda()
    return AutoEncoder

#####加载模型
def autoencoder(model_parameters_path, model_type, dimension, pretrained=False):
    model = select_pretrained_model(model_type, dimension)
    if pretrained:
        model.load_state_dict(torch.load(model_parameters_path))
    return model

def train(model,dataloader,optimizer,loss_func,model_parameters_path,EPOCH):
    model.train()
    for epoch in range(EPOCH):
        total_loss = 0
        train_step = 0
        for mini_data in dataloader:
            optimizer.zero_grad()
            #########forward#########
            encoded, decoded = model(mini_data.cuda())
            loss = loss_func(decoded, mini_data.cuda())
            #########backward########
            loss.backward()
            optimizer.step()
            ##########记录loss的值
            total_loss = total_loss + loss.item()
            train_step += 1
        if (epoch+1) % 5 == 0:    
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1,EPOCH,total_loss/train_step))
        ######保存模型参数
        if (epoch+1) % 5 == 0:
            print("保存模型参数的地址:"+str(model_parameters_path))
            torch.save(model.state_dict(), model_parameters_path)

if __name__=="__main__":
    ######receipt parameters
    parser = argparse.ArgumentParser()
    ######required=True表示必须要输入的变量
    parser.add_argument("--dataset", type=str, help="training dataset", choices=['i','v'], required=True)
    parser.add_argument("--dimension", type=int, choices=[128,256,512,1024,2048,4096], required=True)
    parser.add_argument("--model_type", type=str, choices=['alexnet','vgg16','resnet101','densenet169'])
    parser.add_argument("--EPOCH", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_parameters_path", type=str, required=True)
    args = parser.parse_args()
    ######data address
    dataset_path = args.dataset_path
    model_parameters_path = args.model_parameters_path
    #####模型类型
    model_type = args.model_type
    print("训练数据集地址:"+str(dataset_path))
    print("模型参数地址:"+str(model_parameters_path)) 
    start=time.time()
    #####download dataset
    train_data = load_trained_data(dataset_path, model_type)
    dataloader = DataLoader(train_data, batch_size=256, num_workers=8, shuffle=True)
    #####load model
    model = select_pretrained_model(model_type, args.dimension)
    #####select optimizer method
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #####compute loss function
    loss_func = nn.MSELoss()
    #####train
    train(model, dataloader, optimizer, loss_func, model_parameters_path, args.EPOCH)
    end = time.time()
    print('训练autoencoder模型参数共耗时:'+str(end-start))
