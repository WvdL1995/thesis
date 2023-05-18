import torch
import torch.nn as nn

class opt():
    pass

class DC_Generator_1D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DC_Generator_1D, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(opt.latent_dim,128),
            #nn.BatchNorm1d(1,0.8),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(1,4,3,stride=2,padding=1),
            nn.BatchNorm1d(4,0.8),
            # nn.Flatten(start_dim=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4,8,3,stride=2,padding=1),
            #nn.BatchNorm1d(8,0.8),
            # nn.Flatten(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(8,8,3,stride=2,padding=1),
            nn.Flatten(),
            nn.Linear(512,512),
            #nn.ReLU(),
            nn.Sigmoid()
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
    
class min_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(min_Discriminator,self).__init__()
        self.dense = nn.Linear(512,1)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        return self.activation(self.dense(x))