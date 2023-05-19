# functions

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from tqdm import tqdm

from natsort import natsorted
import os

from models import *

def load_data(path,to3D=True):
    """Loads data into numpy array
    returns either 2d array pixels by mass bins
    or 3d array with pixels by pixels by mass bins"""
    print("Opening "+path)
    data = np.load(path,allow_pickle=True)[()]

    if to3D:
        data = datato3D(data)
    else:
        data = datato2D(data)
    return data

def datato3D(data):
    size = np.append(data['image_shape'][0][0:2],(np.shape(data['intensities'])[1]))
    print("Converting data to image of %d by %d pixels with %d mass bins." % (size[0],size[1],size[2]))
   
    recon = np.zeros(size[:2], dtype=np.float32).reshape(-1)
    recon = np.zeros([np.shape(recon)[0],np.shape(data['intensities'])[1]])

    recon[recon==0] = np.nan
    recon[data['pixel_order']] = data['intensities'][:,:]

    recon = recon.reshape(np.append(data['image_shape'][0][:2],np.shape(recon)[1]))
    return recon

def datato2D(data):
    size = np.append(data['image_shape'][0][0:2],(np.shape(data['intensities'])[1]))
    print("Converting data to array of %d spectra with %d mass bins." % (size[0]*size[1],size[2]))

    recon = np.zeros(size[:2], dtype=np.float32).reshape(-1)
    recon = np.zeros([np.shape(recon)[0],np.shape(data['intensities'])[1]])

    recon[recon==0] = np.nan
    recon[data['pixel_order']] = data['intensities'][:,:]

    # remove nan columns
    recon = recon[~np.isnan(recon).any(axis=1),:]
    print("After removing empty spectra, %d sectra remain." %np.shape(recon)[0])
    return recon

def plot_slice(data,massbin : list):
    """plot slice of 3d data, sliced at masbin"""
    if isinstance(massbin,int):
        plt.imshow(np.rot90(data[:,:,massbin]),origin='lower')
    else:
        #mutliplot implementation
        pass
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Intensity',rotation=270)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

def plot_spect(data,pixel,show=True):
    """Plot spectra, takes two arguments
    data: either 2D or 3D array
    pixel, the pixel of the spectra to plot
    
    if 3D array is given, pixel should be [x,y]
    if multiple pixels are given a subplot is created"""
    if isinstance(pixel,int):
        pixel = [pixel]
    if data.ndim<=2:
        plt.subplots(len(pixel))

        for i in range(len(pixel)):
            plt.subplot(len(pixel),1,i+1)
            plt.plot(data[pixel[i]])                
            #plt.title("Pixel nr. %d" %pixel[i])
    else:
        # print(np.size(pixel))
        if np.size(pixel)>3:
            pass #subplots       
        else:
            spec = data[pixel[0],pixel[1],:]
            plt.plot(spec)
            # plt.ylabel('Intensity')
    plt.ylabel('Intensity')
    plt.xlabel('Mass bin')
    plt.suptitle("Selected spectra")

    if show:
        plt.show()
    return plt

class dataloader(torch.utils.data.TensorDataset):
    def __init__(self,samples,labels):
        self.data=[]
        for i in range(len(labels)):
            self.data.append([samples[:,:,i],labels[i]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample, classname = self.data[idx]
        sample = torch.from_numpy(sample)
        return sample, classname
    
def train(opt,data,generator,discriminator,optimizers,adverloss,savemodels=False):
    Tensor = torch.FloatTensor # no gpu implementation yet
    if savemodels:
        # create nescesary folders if they don't exist!!
        pass
    for epoch in range(opt.n_epochs):
        with tqdm(data,unit="batch") as tepoch:
            tepoch.set_description("Epoch %d / %d" %(epoch+1,opt.n_epochs))
            for sample, _ in tepoch:
                # create real/fake labels
                valid = Variable(Tensor(sample.shape[0],1,1).fill_(1.0),requires_grad=False)
                fake =  Variable(Tensor(sample.shape[0],1,1).fill_(0.0),requires_grad=False)

                real_samples = Variable(sample.type(Tensor))

                optimizers.optimizer_G.zero_grad()

                z = Variable(Tensor(torch.rand((sample.shape[0],1,opt.latent_dim))))
                # print("sample shape:",sample.shape[0])
                # print("noise shape:",np.shape(z))
                gen_samples = generator(z)
                g_loss = adverloss(discriminator(gen_samples),valid)

                g_loss.backward()
                optimizers.optimizer_G.step()
                optimizers.optimizer_D.zero_grad()

                real_loss = adverloss(discriminator(real_samples),valid)
                fake_loss = adverloss(discriminator(gen_samples.detach()), fake) 

                d_loss = (real_loss + fake_loss)/2

                d_loss.backward()
                optimizers.optimizer_D.step()
                tepoch.set_postfix(Gloss=g_loss.item(),Dloss=d_loss.item())
            if savemodels:
                torch.save(generator,str("models/run180523_2/generator"+str(epoch)))
                torch.save(discriminator,str("models/run180523_2/discriminator"+str(epoch)))

def eval_model(model_path,staticnoise,gen_img=False,losses=False):
    """function to evaluate saved models
    Can generate images and calculate losses"""
    gen_model = DC_Generator_1D()
    dis_model = DC_Discriminator_1D()
    H = {'g_loss': [],
         'd_loss': []}
    for filename in natsorted(os.listdir(model_path)):
        if 'generator' in filename:
            fgen = os.path.join(model_path, filename)
            nr = filename.replace('generator','')
            gen_model = torch.load(fgen)

            if losses:
                fdis = os.path.join(model_path, filename.replace('generator','discriminator'))
                print(fdis)
                dis_model = torch.load(fdis)
                ##
                # evaluate loss function
                ##
            if gen_img:
                xhat = gen_model.forward(staticnoise).cpu().detach().numpy()
                imgname = str('fake/img'+str(nr))
                xhat = np.squeeze(xhat,axis=1)
                plt = plot_spect(xhat,([0,1,2,3,4]),show=False)
                plt.savefig(str(model_path+imgname))
        else:
            pass