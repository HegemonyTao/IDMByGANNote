import os # saving files
import numpy as np # for matrix math
import matplotlib.pyplot as plt # plot and visualizing data
from data_utils import shuffle, iter_data # analyzing data
from tqdm import tqdm # progress bar
import preprocessing
import post_processing
import networks as net
import torch
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
#读取数据
NiH = preprocessing.read_data('data/NiH_dataset.mat')
PdH = preprocessing.read_data('data/PdH_dataset.mat')
max_PdH = preprocessing.compute_max(PdH)
max_NiH = preprocessing.compute_max(NiH)
max_len=max(max_PdH,max_NiH)
Ni_H = preprocessing.data_padding(NiH,max_len)
Pd_H = preprocessing.data_padding(PdH,max_len)
#定义超参数
n_epoch =100
batch_size  = 35
input_dim = 3
latent_dim = 3
eps_dim = 3

# 判别器
n_layer_disc = 5
n_hidden_disc = 100

# 生成器
n_layer_gen = 5
n_hidden_gen= 100

# 另一个生成器
n_layer_inf = 5
n_hidden_inf= 100
#创建结果文件夹
result_dir = 'STEP1+STEP2_CrsytalGAN/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)
# AH和BH的真实样本
# 在本例中是Pd_H和Ni_H
AH_dataset = Pd_H
BH_dataset = Ni_H
from Dataset import AHBDataset
from torch.utils.data import DataLoader
dataSet=AHBDataset(AH_dataset,BH_dataset)
dataLoader=DataLoader(dataSet,batch_size=batch_size,shuffle=True)
AHB=net.Generator_AHB(input_dim,n_layer_gen,n_hidden_gen,eps_dim,None)
#AHBOpt=torch.optim.Adam(AHB.parameters(),lr=1e-4,betas=(0.5,0.999))
if is_cuda:
    AHB.cuda()
AHB.train()
AH_dataset, BH_dataset= shuffle(AH_dataset, BH_dataset)
for x,y in iter_data(AH_dataset,BH_dataset,size=batch_size):
    print(np.array(x).shape)
    print(np.array(y).shape)
    print('-------------------')

'''
for data,target in dataLoader:
    if is_cuda:
        data,target=data.cuda(),target.cuda()
    output=AHB(data)
'''