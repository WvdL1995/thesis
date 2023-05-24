import torch
import torch.nn as nn

class opt():
    """Options"""
    pass

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) #mean 0, std 0.02
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) 
        torch.nn.init.constant_(m.bias.data, 0.0)

class DC_Generator_1D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DC_Generator_1D, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(opt.latent_dim,128),
            nn.BatchNorm1d(1,0.8),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(1,16,3,stride=2,padding=1),
            nn.BatchNorm1d(16,0.8),
            # nn.Flatten(start_dim=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16,16,3,stride=2,padding=1),
            nn.BatchNorm1d(16,0.8),
            # nn.Flatten(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(16,8,3,stride=2,padding=1),
            nn.Flatten(),
            nn.Linear(512,512),
            nn.ReLU(),
            #nn.Sigmoid()
            )

    def forward(self,x):
        output = self.layers(x)
        output = output.view(output.shape[0],1,512) #adds additional d
        return output

class DC_Discriminator_1D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DC_Discriminator_1D, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv1d(1,1,3,stride=5,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(1,1,3,stride=5,padding=1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1,0.8),
            nn.Conv1d(1,1,3,stride=4,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1,0.8),
            nn.Linear(6,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.layers(x)

class OG_Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(OG_Generator,self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(opt.latent_dim,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(1,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(1,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.BatchNorm1d(1,0.8),
            nn.LeakyReLU(0.2,inplace=True),   
            nn.Linear(512,512),
            nn.ReLU()        
        )
    def forward(self,x):
        output = self.layers(x)
        output = output.view(output.shape[0],1,512) #adds additional d
        return output

class OG_Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(OG_Discriminator,self).__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        #x = x.view(x.size(0),-1)
        return self.layers(x)

class DC_Discriminator_2D(nn.Module):
    def __init__(self) -> None:
        super(DC_Discriminator_2D,self).__init__()

class min_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(min_Discriminator,self).__init__()
        self.dense = nn.Linear(512,1)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        return self.activation(self.dense(x))
