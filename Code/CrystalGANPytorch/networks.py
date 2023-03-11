import torch
import torch.nn as nn
import torch.nn.functional as F
class Generator_AHB(nn.Module):
    def __init__(self,input_dim,n_layer,n_hidden):
        super(Generator_AHB, self).__init__()
        self.model=nn.Sequential(nn.Linear(input_dim,n_hidden))
        for i in range(n_layer-1):
            self.model.add_module('full_layer'+str(i+1),nn.Linear(n_hidden,n_hidden))
            self.model.add_module('relu'+str(i+1),nn.ReLU())
        self.model.add_module('output',nn.Linear(n_hidden,input_dim))
    def forward(self,x):
        return self.model(x)
class Generator_BHA(nn.Module):
    def __init__(self,latent_dim,n_layer,n_hidden):
        super(Generator_BHA, self).__init__()
        self.model=nn.Sequential(nn.Linear(latent_dim,n_hidden))
        for i in range(n_layer-1):
            self.model.add_module('full_layer'+str(i+1),nn.Linear(n_hidden,n_hidden))
            self.model.add_module('relu'+str(i+1),nn.ReLU())
        self.model.add_module('output',nn.Linear(n_hidden,latent_dim))
    def forward(self,x):
        return self.model(x)
class Discriminator_AH(nn.Module):
    def __init__(self,input_dim,n_layers=2, n_hidden=10):
        super(Discriminator_AH, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, n_hidden))
        for i in range(n_layers-1):
            self.model.add_module('full_layer'+str(i+1),nn.Linear(n_hidden,n_hidden))
            self.model.add_module('relu'+str(i+1),nn.ReLU())
        self.model.add_module('output',nn.Linear(n_hidden,1))
    def forward(self,x):
        x=torch.cat(x,dim=1)
        x=self.model(x)
        x=torch.squeeze(x)
        return F.softplus(x)
class Discriminator_BH(nn.Module):
    def __init__(self,input_dim,n_layers=2,n_hidden=10):
        super(Discriminator_BH, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, n_hidden))
        for i in range(n_layers - 1):
            self.model.add_module('full_layer' + str(i + 1), nn.Linear(n_hidden, n_hidden))
            self.model.add_module('relu' + str(i + 1), nn.ReLU())
        self.model.add_module('output', nn.Linear(n_hidden, 1))
    def forward(self,x):
        #print(x.shape)
        #x=nn.Flatten()(x)
        #x=torch.cat([x],dim=1)
        #print(x.shape)
        x=self.model(x)
        x=torch.squeeze(x)
        return F.softplus(x)
