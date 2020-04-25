import torch
import torch.nn as nn
import torch.nn.functional as F

from net.subnet.discriminator import Discriminator

class CVAE(nn.Module):

    def __init__(self, in_channels, T, n_z, num_class):

        super().__init__()

        self.T   = T
        self.n_z = n_z
        self.num_class = num_class
        
        self.encoder = Encoder(T, in_channels, num_class)
        self.decoder = Decoder(T, in_channels, num_class)

        self.y_discriminator   = Discriminator(num_class)
        self.z_discriminator   = Discriminator(num_class)

    def forward(self, x):

        N, M, T, VC  = x.size()

        mean, logvar = self.encoder(x)
        mean = F.softmax(mean, dim=1 )  

        z = self.reparameter(mean.repeat(M, 1), logvar )
        z = z.view(N, M, -1)

        recon_x = self.decoder(z, T)

        return recon_x, mean, logvar, z

    def reparameter(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn_like(logvar)

        return mean + (eps * std)

    def inference(self, n=1, class_label = [0] ):
        
        batch_size = n

        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.num_class )))
        
        if(self.is_cuda):
            z = z.cuda()
        
        recon_x = self.decoder(z)

        return recon_x


class Encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, T, in_channels, n_z):
        super().__init__()

        self.data_bn = nn.BatchNorm1d(in_channels)

        self.lstm    = nn.ModuleList((
            nn.LSTM(in_channels, 258, 3),
            nn.LSTM(258, 128, 3)
        ))

        # fcn for encoding
        self.z_mean   = nn.Conv2d(T*128, n_z, kernel_size=1)
        self.z_logvar = nn.Conv2d(T*128, n_z, kernel_size=1)

    def forward(self, x):
        
        N,M,T,F = x.size()

        x = x.view(N*M,T,F)

        # data normalization
        x = x.permute(0, 2, 1).contiguous()
        x = self.data_bn(x)
        x = x.permute(2, 0, 1).contiguous()

        # forward
        for layer in self.lstm:
            x, _ = layer(x)
                
        x = x.permute(1,0,2).contiguous()

        # prediction
        mean = x.view( N, M  , x.shape[1] * x.shape[2], 1, 1).mean(dim = 1)

        mean = self.z_mean(mean)
        mean = mean.view(mean.size(0), -1)
        
        
        logvar = x.view( N * M , x.shape[1] * x.shape[2], 1, 1)

        logvar = self.z_logvar(logvar)
        logvar = logvar.view(logvar.size(0), -1)

        return mean, logvar


class Decoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, T, in_channels, n_z):
        super().__init__()

        # build networks
        self.fcn = nn.Sequential(
            nn.BatchNorm2d(n_z ),
            nn.ConvTranspose2d(n_z, T * 128, kernel_size= 1),
            nn.BatchNorm2d(T * 128)
        )
        self.lstm = nn.ModuleList((
            nn.LSTM(128, 258, 3),
            nn.LSTM(258, in_channels, 3)
        ))

        self.data_bn = nn.BatchNorm1d(in_channels)
        self.out     = nn.Sigmoid()

    def forward(self, z,T):
        
        N, M, n_z = z.size()

        # reshape
        z = z.view(N * M, n_z, 1, 1)

        # forward
        z = self.fcn(z)
    
        z = z.view(T, z.shape[0], int(z.shape[1]/T))
        
        # forward
        for layer in self.lstm:
            z, _ = layer(z)

        # data batch normalization
        z = z.permute(1, 2, 0).contiguous()
        z = self.data_bn(z)
        z = z.permute(0, 2, 1).contiguous()
        
        z = z.view(N, M, z.shape[1], z.shape[2])

        return z
