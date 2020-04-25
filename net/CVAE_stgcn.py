import sys
sys.path.extend(['../'])

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph   import Graph
from net.subnet.st_gcn import *
from net.subnet.discriminator import Discriminator


class CVAE(nn.Module):

    def __init__(self, in_channels, T, V, num_class, graph_args,
                 edge_importance_weighting=False, **kwargs):

        super().__init__()
        temporal_kernel_size= [299,299]
        
        self.T = T
        self.V = V
        self.num_class = num_class

        self.encoder = Encoder( in_channels, num_class, 
                                graph_args , edge_importance_weighting,
                                temporal_kernel_size[0]
                                )
        self.decoder = Decoder( in_channels, num_class, self.T, self.V, 
                                graph_args , edge_importance_weighting,
                                temporal_kernel_size[1]
                                )
        
        self.y_discriminator   = Discriminator(num_class)
        self.z_discriminator   = Discriminator(num_class)

    def forward(self, x ):

        N,C,T,V,M = x.size()
        
        # encoder
        cat_y, latent_z = self.encoder(x)

        # Catogerise 
        cat_y = F.softmax( cat_y , dim=1 )

        # Reparameter
        z = self.reparameter(cat_y.repeat(M, 1), latent_z)
        z = z.view(N, M, -1)
        

        recon_x = self.decoder(z)

        return recon_x, cat_y, latent_z, z
    
    
    def reparameter(self, mean, logvar):

        std = torch.exp(0.5 * logvar)    
        eps = torch.randn_like(logvar)

        return mean + eps*std

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

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting = False, 
                 temporal_kernel_size = 9, **kwargs):
        
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A          = torch.tensor(self.graph.A, dtype = torch.float32, 
                                    requires_grad = False)
        
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.encoder = nn.ModuleList((
            st_gcn(in_channels  , 64    , kernel_size, 1, **kwargs),
            st_gcn(64           , 128   , kernel_size, 1, **kwargs),
            st_gcn(128          , 128   , kernel_size, 1, **kwargs)
            
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.encoder
            ])
        else:
            self.edge_importance = [1] * len(self.encoder)

        # fcn for encoding
        self.z_mean     = nn.Conv2d(128, num_class, kernel_size=1)
        self.z_logvar   = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        
        #Data Norm
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        
        # forward
        for gcn, importance in zip(self.encoder, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        
        # prediction
        mean = x.view(N, M, -1, 1 ,1).mean(dim = 1)
        
        mean = self.z_mean(mean)
        mean = mean.view(mean.size(0), -1)

        #latent value
        logvar = x.view(N*M, -1, 1, 1)
        
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

    def __init__(self, in_channels, num_class, T, V, 
                    graph_args, edge_importance_weighting = False, 
                    temporal_kernel_size = 9, **kwargs):
        
        super().__init__()

        # load graph
        self.graph  = Graph(**graph_args)
        A           = torch.tensor(self.graph.A, dtype = torch.float32, 
                                    requires_grad = False)
        
        self.register_buffer('A', A)

        # build networks      
        spatial_kernel_size = A.size(0)
        kernel_size         = (temporal_kernel_size, spatial_kernel_size)


        self.fcn    = nn.ConvTranspose2d(num_class, 128, kernel_size=(T,V))        
        self.fcn_bn = nn.BatchNorm1d(128 * A.size(1))

        self.decoder = nn.ModuleList((
            st_gctn(128 , 128        , kernel_size, 1, **kwargs),
            st_gctn(128 , 64         , kernel_size, 1, **kwargs),
            st_gctn(64  , in_channels, kernel_size, 1, ** kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.decoder
            ])
        else:
            self.edge_importance = [1] * len(self.decoder)

        #ouput Norm
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.out     = nn.Tanh()
        
    def forward(self, z):

        N,M,_ = z.size()

        z = z.view( N * M, -1, 1, 1 )
        
        # Deconvo spatial temporal
        z = self.fcn(z)
        
        _,C,T, V = z.size()

        # z norm
        z = z.permute( 0, 3, 1, 2 ).contiguous() 
        z = z.view(N * M, V * C, T)
        
        z = self.fcn_bn(z) 

        z = z.view(N, M, V, C, T)
        z = z.permute(0, 1, 3, 4, 2).contiguous()
        z = z.view(N * M, C, T, V)

        # Deconvolution forward
        for gcn, importance in zip(self.decoder, self.edge_importance):
            z, _ = gcn(z, self.A * importance)

        # data normalization        
        _, C, T, V,  = z.size()
        
        # output norm
        z = z.view(N, M, C, T, V ).contiguous()
        z = z.permute(0, 1, 4, 2, 3).contiguous()

        z = z.view(N * M, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, M, V, C, T)
        
        z = z.permute(0, 3, 4, 2, 1).contiguous()
        # z = self.out(z)

        return z


if __name__ == '__main__':
    
    x=torch.randn(36,3,300,25,2).cuda()
    
    N, C, T, V, M = x.size()

    graph_args = {"layout":'ntu-rgb+d','strategy': "uniform"}
    
    m = CVAE(in_channels = 3, T = T, V = V, n_z = 32, 
                graph_args = graph_args,
                edge_importance_weighting = True 
                ).cuda()
    
    optimizer   = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    lossF       = nn.MSELoss()
    
    for i in range(10000):

        recon_x, mean, lsig, z = m(x)
        
        optimizer.zero_grad()
        loss = lossF(x, recon_x) 
        
        optimizer.step()
        if (i % 100)==0:
            print(i," : ", loss.item())

    print(recon_x.shape)
    print(mean.shape)
    print(lsig.shape)
    print(z.shape)
    print(lossF(x, recon_x ))
