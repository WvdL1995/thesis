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
    if data.ndim<=2: #data is in 2D
        
        plt.subplots(len(pixel),sharex=True)
        plt.subplots_adjust(hspace=0)

        for i in range(len(pixel)):
            plt.subplot(len(pixel),1,i+1)
            plt.stem(data[pixel[i]],basefmt =' ',markerfmt=' ')
            plt.ylabel('Intensity')

            # if opt.pltlog == True:
            #     plt.yscale('log')   
            #plt.title("Pixel nr. %d" %pixel[i])
    else: #data is in 3D, every pixel needs coordinate
        # print(np.size(pixel))
        if np.size(pixel)>3:
            pass #subplots       
        else:
            spec = data[pixel[0],pixel[1],:]
            plt.stem(spec,markerfmt=' ')
            # plt.ylabel('Intensity')
    
    plt.xlabel('Mass bin')
    title = "Selected spectra:\n"+", ".join([str(i) for i in pixel])
    # print(title)
    plt.suptitle(title)

    if show:
        plt.show()
    return plt
def get_labels(path):
    with np.load(path,allow_pickle=True) as nmfdata:
        w = nmfdata['w']
        h = nmfdata['h']
    classes = []
    for i in range(len(w)):
        classes.append(np.argmax(w[i]))
    return classes

class dataloader(torch.utils.data.TensorDataset):
    def __init__(self,samples,labels):
        self.data=[]
        for i in range(len(labels)):
            self.data.append([samples[:,:,i],labels[i]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample, classname = self.data[idx]
        if torch.cuda.is_available():
            sample = torch.from_numpy(sample).to(torch.device("cuda"))
        else:
            sample = torch.from_numpy(sample)
        return sample, classname
    
def train(opt,data,generator,discriminator,optimizers,adverloss,savedir=False):
    """
    If no savedir is specified, model will not be saved
    """
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    if savedir:
        folder = savedir
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("new directroy created called: %s" %folder)

    for epoch in range(opt.n_epochs):
        with tqdm(data,unit="batch") as tepoch:
            tepoch.set_description("Epoch %d / %d" %(epoch+1,opt.n_epochs))
            for sample, _ in tepoch:
                # create real/fake labels
                valid = Variable(Tensor(sample.shape[0],1,1).fill_(1.0),requires_grad=False)
                fake =  Variable(Tensor(sample.shape[0],1,1).fill_(0.0),requires_grad=False)

                real_samples = Variable(sample.type(Tensor))

                optimizers.optimizer_G.zero_grad()

                if torch.cuda.is_available():
                    z = Variable(Tensor(torch.rand((sample.shape[0],1,opt.latent_dim)).to(torch.device('cuda'))))
                else:
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
                if d_loss.item()>=50 and epoch>=2:
                    count+=1
                    if count>=5:
                        print("_Discriminator loss equal to %d,too large, stopping training...",d_loss.item())
                        return
                else:
                    count = 0
                d_loss.backward()
                optimizers.optimizer_D.step()
                tepoch.set_postfix(Gloss=g_loss.item(),Dloss=d_loss.item())
            if savedir:
                torch.save(generator,str(folder+"generator"+str(epoch)))
                torch.save(discriminator,str(folder+"discriminator"+str(epoch)))

def eval_model(model_path,staticnoise,x_test,gen_img=False,losses=False):
    """function to evaluate saved models
    Can generate images and calculate losses"""
    gen_model = DC_Generator_1D()
    dis_model = DC_Discriminator_1D()
    if torch.cuda.is_available():
        gen_model.cuda()
        dis_model.cuda()
    H = {'g_loss': [],
         'd_loss': [],
         'l1_norm': [],
         'l2_norm': [],}
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    valid = Variable(Tensor(staticnoise.shape[0],1,1).fill_(1.0),requires_grad=False)
    fake =  Variable(Tensor(staticnoise.shape[0],1,1).fill_(0.0),requires_grad=False)

    adverloss = torch.torch.nn.BCELoss()

    for filename in natsorted(os.listdir(model_path)):
        if 'generator' in filename:
            fgen = os.path.join(model_path, filename)
            nr = filename.replace('generator','')
            gen_model = torch.load(fgen)

            if losses:
                fdis = os.path.join(model_path, filename.replace('generator','discriminator'))
                dis_model = torch.load(fdis)
                ##
                gen_samples = gen_model(staticnoise)
                g_loss = adverloss(dis_model(gen_samples),valid)
                if torch.cuda.is_available():
                    real_loss = adverloss(dis_model(torch.Tensor(x_test[0:staticnoise.shape[0],:,:]).to(torch.device('cuda'))),valid)
                else:
                    real_loss = adverloss(dis_model(torch.Tensor(x_test[0:staticnoise.shape[0],:,:])),valid)

                fake_loss = adverloss(dis_model(gen_samples.detach()), fake) 
                d_loss = (real_loss + fake_loss)/2

                H["g_loss"].append(g_loss.item())
                H["d_loss"].append(d_loss.item())

                l1 = evaluation_metrics.l1norm(real=gen_samples.detach().cpu(),fake=x_test[0:staticnoise.shape[0],:,:])
                l2 = evaluation_metrics.l2norm(real=gen_samples.detach().cpu(),fake=x_test[0:staticnoise.shape[0],:,:])
                H["l1_norm"].append(l1)
                H["l2_norm"].append(l2)
                ##
            if gen_img:
                if not os.path.exists(str(model_path+"fake/")):
                    os.makedirs(model_path+"fake/")    
                xhat = gen_model.forward(staticnoise).cpu().detach().numpy()
                imgname = str('fake/img'+str(nr))
                xhat = np.squeeze(xhat,axis=1)
                fig = plot_spect(xhat,([0,1,2,3,4]),show=False)
                fig.savefig(str(model_path+imgname))
                fig.close()
        else:
            pass
    if losses:
        # print("Generator loss",H["g_loss"])
        # print("Discriminator loss",H["d_loss"])
        plt.figure()
        plt.plot(H["g_loss"])
        plt.plot(H["d_loss"])
        plt.legend(["Generator","Discriminator"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("BCE loss on test set")
        plt.savefig(str(model_path+"BCE_testloss"))
        
        plt.figure()
        # plt.subplots(2)
        # plt.subplot(2,1,1)
        # plt.plot(H["l1_norm"])
        # plt.ylabel("Avarage l1 norm")
        # plt.title("Avarage norms over iterations")
        # plt.subplot(2,1,2)
        plt.plot(H["l2_norm"])
        # # plt.legend(["l1 Norm","l2 Norm"])
        plt.xlabel("Epoch")
        plt.ylabel("Avarage l2 norm")
        plt.savefig(str(model_path+"normeval"))

class evaluation_metrics():
    def __init__(self) -> None:
        pass

    def l1norm(fake,real):
        """Calculates avarage l1 norm"""
        # e = real[:len(fake)-1,:,:]-fake
        e = real-fake

        l1 = np.linalg.norm(e,ord=1,axis=0)
        av_l1=np.mean(l1)

        #eb = real[len(fake):2*len(fake),:,:]-fake
        #l1_base = np.linalg.norm(eb,ord=1,axis=0)
        av_l1_base = np.mean(l1)
        return av_l1 ,av_l1_base
    
    def l2norm(real,fake):
        """Calculates average l2 norm"""
        e = real-fake
        l1 = np.linalg.norm(e,ord=2,axis=0)
        av_l1=np.mean(l1)
        return av_l1