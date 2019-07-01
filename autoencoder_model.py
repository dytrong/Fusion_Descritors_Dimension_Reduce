import torch
from torch import nn

class L2Norm(nn.Module):

    def __init__(self):

        super(L2Norm, self).__init__()

        self.eps = 1e-10

    def forward(self, x):

        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)

        x= x / norm.unsqueeze(-1).expand_as(x)

        return x

####auto-encoder model
###20190608 1152->512->256->128, 效果最佳
###20190611 1152->384->128 
###20190613 1152->784->256-128

class autoencoder_cnn_1152(nn.Module):

    def __init__(self, dimension):

        super(autoencoder_cnn_1152, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1152, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512, affine=False),
            nn.PReLU(),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256, affine=False),
            nn.PReLU(),

            nn.Conv2d(256, dimension, 1, stride=1, padding=0),
            nn.BatchNorm2d(dimension, affine=False),
            nn.PReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256, affine=False),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512, affine=False),
            nn.PReLU(),

            nn.ConvTranspose2d(512, 1152, 1, stride=1, padding=0),
            nn.BatchNorm2d(1152, affine=False),
            nn.PReLU(),
        )        
        
    def forward(self,x):

        encoded=self.encoder(x)

        decoded=self.decoder(encoded)

        return encoded.view(x.size(0), -1), decoded

         

###auto-encoder model
class autoencoder_cnn_2176(nn.Module):

    def __init__(self, dimension):

        super(autoencoder_cnn_2176, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2176, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, dimension, 1, stride=1, padding=0),
            nn.BatchNorm2d(dimension),
            nn.PReLU(),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.ConvTranspose2d(512, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.ConvTranspose2d(1024, 2176, 1, stride=1, padding=0),
            nn.BatchNorm2d(2176),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded=self.encoder(x)

        decoded=self.decoder(encoded)

        return encoded.view(x.size(0), -1), decoded

####auto-encoder model
class autoencoder_sigmoid(nn.Module):
    def __init__(self, dimension):
        super(autoencoder_sigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.Linear(256, dimension),
            nn.BatchNorm1d(dimension),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            
            nn.Linear(512, 1152),
            nn.BatchNorm1d(1152),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded = self.encoder(x) 

        decoded = self.decoder(encoded)

        return encoded, decoded


####auto-encoder model
class autoencoder_sigmoid_2176(nn.Module):
    def __init__(self, dimension):
        super(autoencoder_sigmoid_2176, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2176, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.Linear(256, dimension),
            nn.BatchNorm1d(dimension),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 2176),
            nn.BatchNorm1d(2176),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return encoded, decoded

####auto-encoder model
class autoencoder_sigmoid_4224(nn.Module):
    def __init__(self, dimension):
        super(autoencoder_sigmoid_4224, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4224, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.Linear(256, dimension),
            nn.BatchNorm1d(dimension),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 4224),
            nn.BatchNorm1d(4224),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return encoded, decoded

####auto-encoder model
class autoencoder_tanh(nn.Module):

    def __init__(self, dimension):

        super(autoencoder_tanh, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            nn.Linear(256, dimension),
            nn.BatchNorm1d(dimension),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 1152),
            nn.BatchNorm1d(1152),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return encoded, decoded

