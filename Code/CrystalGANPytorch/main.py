import os # saving files
import numpy as np # for matrix math
from tqdm import tqdm # progress bar
import preprocessing
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Dataset import AHBDataset
from torch.utils.data import DataLoader
import networks as net
import torch
import matplotlib.pyplot as plt
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
#读取数据
NiH = preprocessing.read_data('data/NiH_dataset.mat')
PdH = preprocessing.read_data('data/PdH_dataset.mat')
#获取每种元素的最大行数
max_PdH = preprocessing.compute_max(PdH)
max_NiH = preprocessing.compute_max(NiH)
max_len=max(max_PdH,max_NiH)
#做数据填充（0）
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
dataSet=AHBDataset(AH_dataset,BH_dataset)
dataLoader=DataLoader(dataSet,batch_size=batch_size,shuffle=True)
#定义模型
GenAHB=net.Generator_AHB(input_dim,n_layer_gen,n_hidden_gen)
GenBHA=net.Generator_BHA(latent_dim,n_layer_inf,n_hidden_inf)
DisAH=net.Discriminator_AH(input_dim,n_layer_disc,n_hidden_disc)
DisBH=net.Discriminator_BH(input_dim,n_layer_disc,n_hidden_disc)
if is_cuda:
    GenAHB.cuda()
    GenBHA.cuda()
    DisAH.cuda()
    DisBH.cuda()
GenAHB.train()
GenBHA.train()
DisAH.train()
DisBH.train()
#定义优化器
train_gen_op=optim.Adam(params=list(GenAHB.parameters())+list(GenBHA.parameters()),lr=1e-4,betas=[0.5,0.999])
train_disc_op=optim.Adam(params=list(DisAH.parameters())+list(DisBH.parameters()),lr=1e-4,betas=[0.5,0.999])
FG = []
FD = []
for epoch in range(n_epoch):
    print('----------epoch:{}----------'.format(epoch))
    for AH,BH in dataLoader:
        def getLoss(AH,BH):
            if is_cuda:
                AH,BH=AH.cuda(),BH.cuda()
            AH,BH=Variable(AH),Variable(BH)
            AHB=GenAHB(BH)
            BHA=GenBHA(AH)
            rec_AH=GenBHA(AHB)
            rec_BH=GenAHB(BHA)
            encoder_sigmoid_AH=DisBH(AH)
            decorator_sigmoid_AH=DisBH(AHB)
            encoder_sigmoid_BH=DisBH(BH)
            decorator_sigmoid_BH=DisBH(BHA)
            decoder_loss=decorator_sigmoid_AH+decorator_sigmoid_BH
            encode_loss=encoder_sigmoid_AH+encoder_sigmoid_BH
            disc_loss=torch.mean(encode_loss)-torch.mean(decoder_loss)
            cost_AH=torch.mean(torch.pow(rec_AH-BH,2))
            cost_BH=torch.mean(torch.pow(rec_BH-AH,2))
            adv_loss=torch.mean(decoder_loss)
            gen_loss=adv_loss+cost_BH+cost_AH
            return disc_loss,gen_loss,adv_loss,cost_BH,cost_AH
        for i in range(1):
            disc_loss, gen_loss, adv_loss, cost_BH, cost_AH = getLoss(AH, BH)
            train_disc_op.zero_grad()
            disc_loss.backward()
            train_disc_op.step()
        FD.append(disc_loss)
        for i in range(5):
            disc_loss, gen_loss, adv_loss, cost_BH, cost_AH = getLoss(AH, BH)
            train_gen_op.zero_grad()
            gen_loss.backward()
            train_gen_op.step()
        FG.append([adv_loss, cost_BH, cost_AH])
    print('生成器损失：{}'.format(gen_loss))
    print('判别器损失：{}'.format(disc_loss))
#生成数据
n_viz = 1
#BHA_gen = np.array([]); recon_AH = np.array([]); AHB_gen = np.array([]); recon_BH = np.array([]);
GenBHA.eval()
GenAHB.eval()
print(next(GenBHA.parameters()).device)
for _ in range(n_viz):
    for AH,BH in dataLoader:
        if is_cuda:
            AH.cuda()
            BH.cuda()
        AH, BH = Variable(AH,True), Variable(BH,True)
        #生成BHA
        temp_BHA_gen=GenBHA(AH)
        print(temp_BHA_gen)
'''
FG=[item.cpu().detach() for item in FG]
FD=[item.cpu().detach() for item in FD]

print(FD)
print(FG)

fig_curve, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.plot(FD, label="Discriminator")
ax.plot(np.array(FG)[:,0], label="Generator")
ax.plot(np.array(FG)[:,1], label="Reconstruction AH")
ax.plot(np.array(FG)[:,2], label="Reconstruction BH")
#ax.set_yscale('log')
plt.ylabel('Loss')
plt.xlabel('Iteration')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.axis('on')
plt.show()
'''