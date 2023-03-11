import torch.nn as nn
import torch.nn.functional as F
class Generator_AHB(nn.Module):
    def __init__(self,input_dim,n_layer,n_hidden,eps_dim,reuse_):
        super(Generator_AHB, self).__init__()
        inputSize=[35,4,18,3]
        hiddenSize=[35,4,18,n_hidden]
        self.fc1=nn.Linear(inputSize,hiddenSize)
        self.fc2=nn.Linear(hiddenSize,hiddenSize)
        self.fc3=nn.Linear(hiddenSize,inputSize)
    def forward(self,x):
        tmp=x
        print(x.shape)
        '''
        x=F.relu(self.fc1(x))
        print(x.shape)
        '''
        return x

