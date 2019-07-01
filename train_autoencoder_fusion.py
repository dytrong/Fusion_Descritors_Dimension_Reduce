import torch 
from torch import nn 
import torchvision 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np
import time
import argparse
import os
import sys

sys.path.append('/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/')
from autoencoder_model import autoencoder_sigmoid, autoencoder_sigmoid_2176, autoencoder_sigmoid_4224, autoencoder_tanh
from autoencoder_model import autoencoder_cnn_1152, autoencoder_cnn_2176

###reproducible
torch.manual_seed(1)

#torch.cuda.manual_seed(1)

def normalization(input_data):

    mean_data = input_data.mean(axis=1, keepdims = True)

    std_data = input_data.std(axis=1, keepdims = True)

    print('样本均值为:'+str(mean_data))

    print('样本方差为:'+str(std_data))

    output_data = (input_data-mean_data) / std_data

    return output_data

def change_arr_to_tensor(input_data):

    output_data = torch.from_numpy(input_data)

    output_data = output_data.float()

    if args.autoencoder_type == 'cnn':

        output_data = output_data.view(output_data.size(0), output_data.size(1), 1, 1)

    return output_data

###下载训练数据集,对57张图像中所有特征点进行训练
def load_trained_data(dataset_path):

    all_data = np.load(dataset_path)

    all_data = normalization(all_data)

    #train_data = np.delete(all_data, range(0,len(all_data),5000), 0)
     
    train_data = all_data   
 
    test_data = all_data[::5000,:]

    train_data = change_arr_to_tensor(train_data)

    test_data = change_arr_to_tensor(test_data)

    print('训练autoencoder训练数据集大小为:'+str(train_data.shape))

    print('训练autoencoder测试数据集大小为:'+str(test_data.shape))

    return train_data, test_data

def fusion_autoencoder(model_parameters_path, dimension, autoencoder_type='sigmoid', flag='GPU', **kwargs):
    ####选择不同的autoencoder类型
    if autoencoder_type == 'cnn':

        model = autoencoder_cnn_1152(dimension)
        #model = autoencoder_cnn_2176(dimension)

    if autoencoder_type == 'sigmoid':

        #model = autoencoder_sigmoid(dimension)
        #model = autoencoder_sigmoid_2176(dimension)
        model =  autoencoder_sigmoid_4224(dimension)

    if autoencoder_type == 'tanh':

        model = autoencoder_tanh(dimension)

    ####加载cpu或者gpu模型
    if flag == 'GPU':

        model.load_state_dict(torch.load(model_parameters_path))

    if flag == 'CPU':

        model.load_state_dict(torch.load(model_parameters_path, map_location=lambda storage, loc: storage))

    return model

def fusion_test_autoencoder(model_parameters_path, dimension, autoencoder_type, flag, desc):

    model = fusion_autoencoder(model_parameters_path, 
                               dimension, 
                               autoencoder_type, 
                               flag)
    model.eval()

    desc = torch.from_numpy(desc)

    if autoencoder_type == 'cnn':

        desc = desc.view(desc.size(0), desc.size(1), 1, 1)

    encoder, decoder = model(desc)

    encoder = encoder.cpu().detach().numpy()

    return encoder

def train(model, train_dataloader, test_dataloader, optimizer, loss_func, model_parameters_path, EPOCH):
    
    #model.train()   

    for epoch in range(EPOCH):
 
        model.train()

        #scheduler.step()

        train_loss = 0

        train_step = 0

        for train_mini_data in train_dataloader:

            #########forward#########
            optimizer.zero_grad()

            encoded, decoded = model(train_mini_data.cuda())

            loss = loss_func(decoded, train_mini_data.cuda())

            #########backward########
            loss.backward()

            #######record loss value#####
            train_loss = train_loss + loss.item()

            optimizer.step()

            train_step += 1
        '''    
        ##########eval################################
        model.eval()

        test_loss = 0

        test_step = 0

        for test_mini_data in test_dataloader:

            test_encoded, test_decoded = model(test_mini_data.cuda())

            loss = loss_func(test_decoded, test_mini_data.cuda())

            test_loss = test_loss + loss.item()

            test_step += 1
        '''

        if (epoch+1) % 5 ==  0:

            #print('epoch [{}/{}], train loss:{:.4f}, \teval loss:{:.4f}'.format(epoch+1, EPOCH, train_loss/train_step, test_loss/test_step))
            print('epoch [{}/{}], train loss:{:.4f}'.format(epoch+1, EPOCH, train_loss/train_step))
        
        if (epoch+1) % 50 == 0:

            ######保存模型参数
            model_parameters = str(epoch+1) + '_' + model_parameters_path
 
            para_path = './model_parameters/' + args.autoencoder_type

            model_parameters_file = os.path.join(para_path, args.pre_trained_cnn, args.dataset, model_parameters)

            torch.save(model.state_dict(), model_parameters_file) 

if __name__=="__main__":
    ######接收参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['i', 'v'])
    parser.add_argument("--pre_trained_cnn", type=str, choices=['alexnet','resnet101', 'densenet169'], required=True)
    parser.add_argument("--dimension", type=int, choices=[128,256,512,1024,2048,4096], required=True)
    parser.add_argument("--pre_trained_descs_path", type=str, required=True)
    parser.add_argument("--autoencoder_type", type=str, choices=['cnn', 'sigmoid', 'tanh'])
    parser.add_argument("--autoencoder_parameter_path", type=str, required=True)
    parser.add_argument("--EPOCH", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    
    #####data and model address
    dataset_path = args.pre_trained_descs_path

    model_parameters_path = args.autoencoder_parameter_path

    #####download dataset
    train_data, test_data = load_trained_data(dataset_path)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=True)

    start=time.time()

    ####upload model
    if args.autoencoder_type == 'cnn':

        model = autoencoder_cnn_1152(args.dimension).cuda()
        #model = autoencoder_cnn_2176(args.dimension).cuda()

    if args.autoencoder_type == 'sigmoid':

        #model = autoencoder_sigmoid(args.dimension).cuda()
        #model = autoencoder_sigmoid_2176(args.dimension).cuda()
        model = autoencoder_sigmoid_4224(args.dimension).cuda()

    if args.autoencoder_type == 'tanh':
  
        model = autoencoder_tanh(args.dimension).cuda()

    ####select optimizer method
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    ####decrease learning rate
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)

    ####computer loss function
    loss_func = nn.MSELoss()

    ####train
    #train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_func, model_parameters_path, args.EPOCH)
    train(model, train_dataloader, test_dataloader, optimizer, loss_func, model_parameters_path, args.EPOCH)
    

    end=time.time()

    print('训练autoencoder模型参数共耗时:'+str(end-start))    
