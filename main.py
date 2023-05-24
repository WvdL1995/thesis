#!/usr/bin/env python
# file used to run main program

from utils import *
from models import *
import numpy as np

import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# #create 3d  (contains Nan values)
# data = load_data("data/2022-12-08-rat_kidney.npy")
# # create 2D data
data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)

#initialize variables
opt.latent_dim=100
opt.specsize = 600
opt.n_epochs = 1
opt.b1=0.5
opt.b2=0.999
opt.lr=0.02
opt.bsize = 128
opt.pltlog = True

# normalize data
# maxval = np.max(data2D)

# change datasize to 512 mass bins (to make CNN architecture easier)
data2D = data2D[:,0:512]
randomspec = np.random.randint(0,len(data2D),size=5)
# plot_spect(data2D,randomspec)

data2D = np.expand_dims(data2D,axis=2) # number of input channels has to be included in dimensions
labels = np.zeros([len(data2D),1])
data2D,x_test,labels,y_test = train_test_split(data2D,labels,train_size=0.75)
data = np.transpose(data2D)
data = dataloader(data,labels)
data = torch.utils.data.DataLoader(data,
                                    batch_size=opt.bsize,
                                    num_workers=0,
                                    collate_fn=None)

generator = DC_Generator_1D()
discriminator = DC_Discriminator_1D()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if torch.cuda.is_available:
    generator.cuda()
    discriminator.cuda()


class optimizers():
    pass
optimizers.optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizers.optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

adverloss = torch.nn.BCELoss()

modelname = 'models/run220523_4/'
train(opt,data,generator,discriminator,optimizers,adverloss,savedir=modelname)

# evaluation
if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    staticnoise = Variable(Tensor(torch.rand((100,1,opt.latent_dim)).to(torch.device('cuda'))))

else:
    Tensor = torch.FloatTensor
    staticnoise = Variable(Tensor(torch.rand((100,1,opt.latent_dim))))
x_test=np.transpose(x_test,(0,2,1))
eval_model(modelname,staticnoise,x_test,gen_img=True,losses=True)


