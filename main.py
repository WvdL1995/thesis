#!/usr/bin/env python
# file used to run main program

from utils import *
from models import *
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# #create 3d  (contains Nan values)
# data = load_data("data/2022-12-08-rat_kidney.npy")
# # create 2D data
data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)
print(np.min(data2D))


# # show examples of data
# np.max(data)

# plot_slice(data,1)
# plot_spect(data,([200,200]))

#initialize model
opt.latent_dim=100
opt.specsize = 600
opt.n_epochs = 100
opt.b1=0.5
opt.b2=0.999
opt.lr=0.002
opt.bsize = 128

# normalize data
maxval = np.max(data2D)
data2D = data2D/maxval
print(np.max(data2D))
# change datasize to 512 mass bins (to make CNN architecture easier)
data2D = data2D[:,0:512]
randomspec = np.random.randint(0,len(data2D),size=5)
plot_spect(data2D,randomspec)

data2D = np.expand_dims(data2D,axis=2) # number of input channels has to be included in dimensions
print("expanded array",np.shape(data2D))
labels = np.zeros([len(data2D),1])

print(len(labels))
data2D,_,labels,_ = train_test_split(data2D,labels,train_size=0.75)
print(len(labels))
data = np.transpose(data2D)
data = dataloader(data,labels)
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



# train(opt,data,generator,discriminator,optimizers,adverloss,savemodels=True)

Tensor = torch.FloatTensor # no gpu implementation yet
staticnoise = Variable(Tensor(torch.rand((20,1,opt.latent_dim))))
eval_model('models/run180523_2/',staticnoise,gen_img=True,losses=False)


