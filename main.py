#!/usr/bin/env python
# file used to run main program

from utils import *
from models import *
import numpy as np
from tqdm import tqdm

# #create 3d  (contains Nan values)
# data = load_data("data/2022-12-08-rat_kidney.npy")
# # create 2D data
data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)
print(np.min(data2D))


# # show examples of data
# np.max(data)
# randomspec = np.random.randint(0,len(data2D),size=5)
# plot_spect(data2D,randomspec)
# plot_slice(data,1)
# plot_spect(data,([200,200]))


#initialize model
opt.latent_dim=100
opt.specsize = 600
opt.n_epochs = 10
opt.b1=0.5
opt.b2=0.999
opt.lr=0.002
opt.bsize = 64
opt.n_epcohs = 20


# change datasize to 512 mass bins (to make NN architecture easier)
data2D = data2D[:,0:512]

labels = np.zeros([len(data2D),1])
# split into train test set!!
data = dataloader(np.transpose(data2D),labels)
data = torch.utils.data.DataLoader(data,
                                    batch_size=opt.bsize,
                                    num_workers=0,
                                    collate_fn=None)

generator = DC_Generator_1D()
discriminator = DC_Discriminator_1D()

class optimizers():
    pass
optimizers.optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizers.optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))

adverloss = torch.nn.BCELoss()

# train(opt,data,generator,discriminator,optimizers,adverloss)
# generator.eval()
# discriminator.eval()
Tensor = torch.FloatTensor # no gpu implementation yet
z = Variable(Tensor(torch.rand((opt.bsize,1,opt.latent_dim))))
fakedata = generator.forward(z).detach().numpy()
# #  plot_spect(fakedata[:,:],[1,3])
print(np.shape(fakedata))
print("ff",np.shape(next(iter(data))[0].detach().cpu().numpy()))
discriminator.forward(next(iter(data))[0])
pred = discriminator.forward(generator.forward(z))
# print(np.shape(pred))
