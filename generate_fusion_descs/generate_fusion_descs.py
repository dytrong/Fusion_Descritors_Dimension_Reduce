import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append('../AE/')

from pre_trained_desc import select_pretrained_cnn_model
from compute_distance import compute_cross_correlation_match
from compute_average_precision import compute_AP
from compute_keypoints_patch import * 
from forward import *
import time
from delete_dir import *
import os
import configparser
import argparse
from sklearn import preprocessing

sys.path.append('../../')
from utils.dimension_reduce_method import select_dimension_reduce_model, dimension_reduce_method


#####global variable######
img_to_tensor = transforms.ToTensor()

######初始化参数
config = configparser.ConfigParser()
config.read('./setup.ini')
Model_Img_size = config.getint("DEFAULT", "Model_Image_Size")
Max_kp_num = config.getint("DEFAULT", "Max_kp_num")
img_suffix = config.get("DEFAULT", "img_suffix")
txt_suffix = config.get("DEFAULT", "file_suffic")
Image_data = config.get("DATASET", "Hpatch_Image_Path")


######接收参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['i','v'], required=True)
parser.add_argument("--dimension", type=int, choices=[128,256,512,1024,2048,4096], required=True)
parser.add_argument("--reduce_method", type=str,choices=['PCA', 'RP', 'AUTO', 'Isomap'], required=True)
parser.add_argument("--autoencoder_parameter_path", type=str, required=True)
parser.add_argument("--pre_trained_model", type=str, choices=['alexnet','vgg16','resnet101','densenet169'])
parser.add_argument("--pre_trained_descs_path", type=str, required=True)
parser.add_argument("--fusion_method", type=str, choices=['cat', 'mul', 'sum', 'LR'], required=True)
args = parser.parse_args()

#####download models######
mynet = select_pretrained_cnn_model(args.pre_trained_model)

#####class#########
class generate_des:
    def __init__(self,net,img_tensor,mini_batch_size,net_type=args.pre_trained_model):
        self.descriptor=self.extract_batch_conv_features(net,img_tensor,mini_batch_size,net_type)
    
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size,net_type):
        batch_number = len(input_data)//mini_batch_size
        #####计算第一个块的卷积特征
        descriptor = self.extract_conv_features(net,input_data[:mini_batch_size],net_type).cpu().detach().numpy()
        for i in range(1,batch_number):
            if i < batch_number-1:
                mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            #######计算最后一个块,大于等于mini_batch_size
            if i == batch_number-1:
                mini_batch = input_data[mini_batch_size*i:len(input_data)]
            temp_descriptor = self.extract_conv_features(net,mini_batch,net_type).cpu().detach().numpy()
            #####np.vstack纵向拼接，np.hstack横向拼接
            descriptor = np.vstack((descriptor,temp_descriptor))
        return descriptor

    #####extract conv features#####
    def extract_conv_features(self,net,input_data,net_type):
        if net_type.startswith('alexnet'):
            x = alexnet(net,input_data)
        if net_type.startswith('vgg16'):
            x = vgg16(net,input_data)
        if net_type.startswith('vgg19'):
            x = vgg19(net,input_data)
        if net_type.startswith('inception_v3'):
            x = inception_v3(net,input_data)
        if net_type.startswith('resnet'):
            x = resnet(net,input_data)
        if net_type.startswith('densenet'):
            x = densenet(net,input_data)
        return x

#####change images to tensor#####
def change_images_to_tensor(H5_Patch, norm_flag=False):
    img_list=[]
    #the patch image .h5 file
    Img_h5=h5py.File(H5_Patch,'r')
    for i in range(len(Img_h5)):
        img=Img_h5[str(i)][:]
        ###change image format from cv2 to Image
        img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img=img.resize((Model_Img_size, Model_Img_size))
        img=img_to_tensor(img)
        img=img.numpy()
        img=img.reshape(3, Model_Img_size, Model_Img_size)
        img_list.append(img)
    img_array=np.array(img_list)
    #####将数据转为tensor类型
    img_tensor=torch.from_numpy(img_array)
    #####数据归一化
    if norm_flag:
        img_tensor -= 0.443728476019
        img_tensor /= 0.20197947209
    return img_tensor

def compute_batch_descriptors(Img_path, H5_Patch, mini_batch_size=16):
    ####generate patch img .h5 file and return valid key points
    valid_keypoints = compute_valid_keypoints(Img_path, H5_Patch, Max_kp_num)
    #####H5_patch_path用来保存每张图片的keypoint patches,方便计算CNN描述符
    input_data = change_images_to_tensor(H5_Patch)
    #####计算描述符
    desc = generate_des(mynet, input_data.cuda(), mini_batch_size).descriptor
    return valid_keypoints,desc

#######数据融合
def data_fusion(desc1, desc2, fusion_method):

    ######拼接concatenate((array1,array2),axis=1)表示横向拼接
    if fusion_method == 'cat':

        desc = np.concatenate((desc1, desc2), axis=1)

    if fusion_method == 'AE':
  
        desc = np.concatenate((desc1, desc2), axis=1)

        sys.path.append("/home/data1/daizhuang/pytorch/cnn_code/")

        from fusion_parameters.train_autoencoder_fusion import fusion_test_autoencoder

        AE_model_parameters_path = "/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/model_parameters/sigmoid/densenet169/v/" 

        AE_model_parameters_path = AE_model_parameters_path + "200_4224_train_test_v_densenet169_hardnet_sigmoid_256_64_0.001.pth"

        desc = fusion_test_autoencoder(AE_model_parameters_path, 256, 'sigmoid', 'CPU', desc) 

    print('融合后描述符维度:' + str(desc.shape))

    return desc

#######标准化
def normalization_desc(desc):
    mean_data = desc.mean(axis=1, keepdims=True)
    std_data = desc.std(axis=1, keepdims=True)
    desc -= mean_data
    desc /= std_data
    return desc

def Hardnet_desc(Img_path, H5_Patch, norm_flag = True):
    #####导入HardNet代码路径
    sys.path.append('/home/data1/daizhuang/matlab/hardnet/examples/keypoint_match')
    from run_hardnet import compute_kp_and_desc
    kp, hardnet_desc = compute_kp_and_desc(Img_path, H5_Patch)
    if norm_flag:
        if args.dataset == 'i':
            hardnet_desc -= 0.0001433593
            hardnet_desc /= 0.08838826
        if args.dataset == 'v':
            hardnet_desc -= -0.000108430664
            hardnet_desc /= 0.08838831
    return kp, hardnet_desc

def compute_fusion_model_descriptors(Img_path, H5_Patch, reduce_model, reduce_flag=True, fusion_flag=True):
    
    reduce_method = args.reduce_method

    pre_trained_model_type = args.pre_trained_model

    ######pre-trained CNN descriptor
    kp, desc = compute_batch_descriptors(Img_path, H5_Patch)

    if args.dataset == 'i':

        desc = normalization_desc(desc)

    #####reduce dimension
    if reduce_flag:
        desc = dimension_reduce_method(reduce_model, desc, reduce_method, pre_trained_model_type)

    ####HardNet descriptors
    if fusion_flag:
        kp, hardnet_desc = Hardnet_desc(Img_path, H5_Patch)
        ###descriptor fusion
        desc = data_fusion(desc, hardnet_desc, 'AE')
    return kp, desc

def compute_mAP(file_path,reduce_dimension_model): 
    total_AP = []
    extract_desc_time = []
    compute_desc_dis_time = []
    desc = np.zeros((1, 256), dtype='float32')
    #desc = np.zeros((1, args.dimension+128), dtype = 'float32')
    #####子数据集地址
    base_path = Image_data + str(file_path) + '/'
    for i in range(1,7):
        print("start compute the "+str(i)+" pairs matches") 
        Img_path_B = base_path+str(i)+img_suffix
        H5_Patch_B = './data/h5_patch/img'+str(i)+txt_suffix
        img2 = cv2.imread(Img_path_B)
        #############提取特征点，和卷积描述符
        start = time.time()
        kp2,desc2 = compute_fusion_model_descriptors(Img_path_B, H5_Patch_B, reduce_dimension_model)
        desc = np.vstack((desc, desc2))
        extract_desc_time.append(time.time()-start)
    print('提取描述符平均耗时:'+str(np.mean(extract_desc_time)))
    ####去掉第一行无效数据
    return desc[1:,]

if __name__ == "__main__":
    start = time.time()
    all_mAP = []
    Count = 0
    ######alexnet 30976, resnet101 147456, densenet169, 184320
    desc = np.zeros((1, 256), dtype = 'float32')
    #desc = np.zeros((1, args.dimension+128), dtype = 'float32')
    #####返回降维模型
    reduce_dimension_model = select_dimension_reduce_model(args.reduce_method, 
                                                           args.dimension,
                                                           args.pre_trained_descs_path,
                                                           args.autoencoder_parameter_path,
                                                           args.pre_trained_model
                                                          )
    #####遍历图像数据集
    for roots, dirs, files in os.walk(Image_data):
        for Dir in dirs:
            if Dir[0] == args.dataset:
                print('读取的图像:'+Dir)
                Count = Count+1
                print('读取的图片张数:'+str(Count))
                desc2 = compute_mAP(Dir,reduce_dimension_model)
                desc = np.vstack((desc, desc2))
                print('\n')
    print('总共耗时:'+str(time.time()-start))

    #fusion_desc_path = '/home/data/daizhuang/dataset/all_desc/resnet101/'+str(args.dataset) + '_resnet101_all_descs.npy'
    ###直接将hardnet 和 densenet 拼接起来的描述符
    #fusion_desc_path = '/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/fusion_descriptors/densenet169/' + str(args.dataset) + '_' + str(desc.shape[1]) + '_densenet169_hardnet_all_descs.npy'
    ####通过AE融合后的描述符
    fusion_desc_path = '/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/fusion_descriptors/densenet169/' + str(args.dataset) + '_' + str(desc.shape[1]) + '_AE_fusion_all_descs.npy'
    #fusion_desc_path = '/home/data1/daizhuang/pytorch/cnn_code/AE/data/descriptors/hardnet/v_hardnet_all_descs.npy'
    #fusion_desc_path = '/home/data1/daizhuang/pytorch/cnn_code/AE/data/descriptors/resnet101/reduce_dimension_descs/'+str(args.dataset)+'_'+str(args.dimension) +'_resnet101_all_descs.npy'
    np.save(fusion_desc_path, desc[1:,])
    print(desc.shape)
