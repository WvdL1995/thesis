#!/usr/bin/env python
# file used to run main program

from utils import *
from models import *
import numpy as np
import argparse

import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim",type=int,default=100,help="length of latent vector")
parser.add_argument("--specsize",type=int,default=100,help="????")
parser.add_argument("--n_epcohs",type=int,default=50,help="number of epochs")
parser.add_argument("--b1",type=float,default=0.5,help="?????????")
parser.add_argument("--b2",type=float,default=0.999,help="?????????????")
parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
parser.add_argument("--bsize",type=int,default=32,help="batch size")
parser.add_argument("--pltlog",type=bool,default=False,help="plot spectra on logaritmich scale")
options = parser.parse_args()

# print(parser.parse_args())
# quit()
# opt(options)
# opt = opt(vars(options))
# print(type(opt))
# print(opt.__dict__)
# print(opt.latent_dim)

# opt.latent_dim = 100
# print(opt.latent_dim)
# for key,i in enumerate(vars(options)):
#     print(key,dict[key])
#     opt.key = vars(options)[key]
# print(opt.__dict__)

# #create 3d  (contains Nan values)
# data = load_data("data/2022-12-08-rat_kidney.npy")
# # create 2D data
data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)

#initialize variables


opt.latent_dim=100
opt.specsize = 600
opt.n_epochs = 3
opt.b1=0.5
opt.b2=0.999
opt.lr=0.0001 #need extremly low learning rate!
opt.bsize = 256
opt.pltlog = False

# normalize data
# maxval = np.max(data2D)

# change datasize to 512 mass bins (to make CNN architecture easier)
data2D = data2D[:,0:512]
randomspec = np.random.randint(0,len(data2D),size=5)
# plot_spect(data2D,randomspec)
# quit()
data2D = np.expand_dims(data2D,axis=2) # number of input channels has to be included in dimensions
labels = np.zeros([len(data2D),1])
data2D,x_test,labels,y_test = train_test_split(data2D,labels,train_size=0.75)
data = np.transpose(data2D)
data = dataloader(data,labels)
data = torch.utils.data.DataLoader(data,
                                    batch_size=opt.bsize,
                                    num_workers=0,
                                    collate_fn=None)

generator = OG_Generator()
discriminator = OG_Discriminator()
# discriminator = DC_Discriminator_1D()
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()


class optimizers():
    pass
optimizers.optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizers.optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

adverloss = torch.nn.BCELoss()

modelname = 'models/run290523_7/'
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


