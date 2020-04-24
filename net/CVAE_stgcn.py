import sys
sys.path.extend(['../'])


import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph import Graph
from net.subnet.st_gcn import *
from net.subnet.discriminator import Discriminator
import numpy as np

class CVAE(nn.Module):

    def __init__(self, in_channels, T, V, M, num_class, graph_args,
                 edge_importance_weighting=False, **kwargs):

        super().__init__()
        temporal_kernel_size= [299,299]
        self.T = T
        self.V = V
        self.M = M
        # num_class = 120
        self.num_class = num_class

        self.encoder = Encoder(in_channels, num_class, self.M,  graph_args, edge_importance_weighting,temporal_kernel_size[0])
        # self.fcn = nn.Linear()
        self.decoder = Decoder(in_channels, num_class,self.T,self.V,self.M, graph_args, edge_importance_weighting,temporal_kernel_size[1])
        self.y_discriminator   = Discriminator(num_class)
        self.z_discriminator   = Discriminator(num_class)

    def forward(self, x ):
        
        N,C,T,V,M = x.size()
        
        # mean, lsig
        # for m in range(M): 
            
        #     body = x[:,:,:,:,m].view(N,C,T,V,1)

            # mean, lsig = self.encoder(body)
        
        mean, lsig = self.encoder(x)

        mean = F.softmax(mean,dim=1)

        z = self.reparameter(mean,lsig)
        
        

        recon_x = self.decoder(z)

        return recon_x, mean, lsig, z
    
    # def reparameter(self,batch_size,mean,lsig):
        
    #     sig = torch.exp(0.5 * lsig)
        
    #     eps = to_var(torch.randn([batch_size, self.n_z]))

    #     z = eps * sig + mean
    #     return z
    
    def reparameter(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        
        eps = torch.randn_like(logvar)

        return mu + eps*std

    def inference(self, n=1, ldec=None):

        batch_size = n
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.n_z)))
        
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

    def __init__(self, in_channels, n_z, M, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=9, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)

        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.encoder = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128 , kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs)
            
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.encoder
            ])
        else:
            self.edge_importance = [1] * len(self.encoder)
        
        self.body_fcn = nn.Sequential (
            nn.Conv2d(M, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            )

        # fcn for encoding
        self.z_mean = nn.Conv2d(128, n_z, kernel_size=1)
        
        self.z_lsig = nn.Conv2d(128, n_z, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()

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

        x = x.view(N, M, -1, 1)
        x = self.body_fcn(x)

        x = x.view(N,-1 ,1 ,1)

        # prediction
        mean = self.z_mean(x)
        mean = mean.view(mean.size(0), -1)
        lsig = self.z_lsig(x)
        lsig = lsig.view(lsig.size(0), -1)

        return mean, lsig


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

    def __init__(self, in_channels, n_z,T,V,M, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=9, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)


        self.fcn = nn.ConvTranspose3d(n_z, 128, kernel_size=(M,T,V))
        
        self.fcn_bn = nn.BatchNorm1d(128 * A.size(1))

        self.decoder = nn.ModuleList((
            
            st_gctn(128, 128, kernel_size, 1, **kwargs),
            
            st_gctn(128, 64, kernel_size, 1, **kwargs),
            
            st_gctn(64, in_channels, kernel_size, 1, ** kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.decoder
            ])
        else:
            self.edge_importance = [1] * len(self.decoder)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.out = nn.Tanh()
        
    def forward(self, z):

        N = z.size()[0]
        
        # reshape
        z = z.view(z.size(0), z.size(1), 1, 1, 1)
        
        # z = z.repeat([1, 1, T, V])
        # forward
        z = self.fcn(z)
        
        N, C, M, T, V = z.size()
        
        z = z.permute(0,2,1,4,3).contiguous()
        
        z = z.view(N * M, V * C, T)
        

        z = self.fcn_bn(z)

        z = z.view(N, M, V, C, T)
        z = z.permute(0, 1, 3, 4, 2).contiguous()
        z = z.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.decoder, self.edge_importance):
            z, _ = gcn(z, self.A * importance)

        # z = torch.unsqueeze(z, 4)

        # data normalization
        
        _, C, T, V,  = z.size()
        
        z = z.view(N,M,C,T,V).contiguous()
        
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
    graph_args = {"layout":'ntu-rgb+d','strategy': "uniform", 'max_hop': 1, 'dilation': 1}
    m = CVAE(in_channels=3, T=T, V=V, n_z=32, graph_args= graph_args,edge_importance_weighting=True).cuda()
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    lossF = nn.MSELoss()
    
    for i in range(10000):
        recon_x, mean, lsig, z = m (x)
        loss = lossF(x,recon_x) 
        optimizer.step()
        if (i % 100)==0:
            print(i," : ", loss.item())

    print(recon_x.shape)
    print(mean.shape)
    print(lsig.shape)
    print(z.shape)
    print(lossF(x,recon_x))