#!/usr/bin/env python
# pylint: disable=W0201

import sys
# sys.path.extend(['../'])

import argparse
import yaml
import numpy as np
import time
import itertools
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.autograd import Variable
from sklearn.metrics.cluster import homogeneity_score

# torchlight
import torchlight
from   torchlight import str2bool
from   torchlight import DictAction
from   torchlight import import_class

from .processor import Processor

torch.set_printoptions(threshold=5000)

def weights_init(m):    
    classname = m.__class__.__name__
    
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
        if m.bias is not None:
            m.bias.data.fill_(0)
    
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
        if m.bias is not None:
            m.bias.data.fill_(0)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    def get_homogeneity(self,cat_y,label):
        #get matched
        (values, indices) = cat_y.max(dim=1)
        
        pred_label = indices.view(-1).cpu().data.numpy()
        true_label = label.view(-1).cpu().data.numpy()
        
        return homogeneity_score(true_label, pred_label)
    
    def loss(self,recon_x, x,label, cat_y, logvar):
        
        # args = { "size_average":False,"reduce": True, "reduction" : "sum"}
        args = {"reduction" : "mean"}


        weight = torch.tensor([1, 1, 1, 0, 1],requires_grad=False).to(self.dev)

        N,C,T,V,M = x.size()
        
        #spatial loss & pos
        recon_loss = weight[0] * nn.functional.mse_loss(recon_x, x,**args)

        #velocity loss 
        t1 = x[:, :, 1:]       - x[:, :, :-1]
        t2 = recon_x[:, :, 1:] - recon_x[:, :, :-1]

        recon_loss += weight[1] * nn.functional.mse_loss(t1, t2, **args)

        #acceleration loss
        a1 = x[:, :, 2:]       - 2 * x[:, :, 1:-1]       + x[:, :, :-2]
        a2 = recon_x[:, :, 2:] - 2 * recon_x[:, :, 1:-1] + recon_x[:, :, :-2]

        recon_loss += weight[2] * nn.functional.mse_loss(a1, a2, **args)
        
        #catogory loss(classify loss)
        # cat_loss = F.cross_entropy(cat_y, label, **args )
        # cat_loss = weight[3] * cat_loss

        # Discriminator loss
        valid   = Variable( torch.zeros( cat_y.shape[0] , 1 ).fill_(1.0), requires_grad=False ).float().to(self.dev)
        d_loss  = F.binary_cross_entropy(self.model.y_discriminator(cat_y) , valid, **args )

        valid   = Variable( torch.zeros( logvar.shape[0] , 1 ).fill_(1.0), requires_grad=False ).float().to(self.dev)
        d_loss += F.binary_cross_entropy(self.model.z_discriminator(logvar), valid,**args )
        
        d_loss = weight[4] * d_loss

        
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1).mean()
        
        return (recon_loss + d_loss) / (weight.sum())

    def load_model(self):
        
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)

    def load_optimizer(self):
        
        if( self.arg.optimizer == 'SGD'):
            self.optimizer = dict() 
            
            self.optimizer["autoencoder"] =  optim.SGD(
                itertools.chain(self.model.encoder.parameters(), self.model.parameters()),
                lr           = self.arg.base_lr,
                momentum     = 0.9,
                nesterov     = self.arg.nesterov,
                weight_decay = self.arg.weight_decay
                )

            self.optimizer["y_discriminator"] =  optim.SGD(
                self.model.y_discriminator.parameters(),
                lr           = self.arg.base_lr,
                momentum     = 0.9,
                nesterov     = self.arg.nesterov,
                weight_decay = self.arg.weight_decay
                )

            self.optimizer["z_discriminator"] =  optim.SGD(
                self.model.z_discriminator.parameters(),
                lr           = self.arg.base_lr,
                momentum     = 0.9,
                nesterov     = self.arg.nesterov,
                weight_decay = self.arg.weight_decay
                )

        elif( self.arg.optimizer == 'Adam'):
            self.optimizer = dict()

            self.optimizer["autoencoder"] = optim.Adam(
                itertools.chain(self.model.encoder.parameters(), self.model.parameters()),
                lr           = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
                )

            self.optimizer["y_discriminator"] = optim.Adam(
                self.model.y_discriminator.parameters(),
                lr           = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
                )
            
            self.optimizer["z_discriminator"] = optim.Adam(
                self.model.z_discriminator.parameters(),
                lr           = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
                )
        else:
            raise ValueError()

    def adjust_lr(self):

        if self.arg.step:
            lr = self.arg.base_lr * ( 0.1 ** np.sum( self.meta_info['epoch'] >= np.array(self.arg.step)))
            
            for name, optimizer in self.optimizer.items():
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.lr = lr

    def show_topk(self, k):
        rank        = self.result.argsort()

        hit_top_k   = [ l in rank[i, -k:] for i, l in enumerate(self.label)]
        
        accuracy    = sum(hit_top_k) * 1.0 / len(hit_top_k)
        
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        self.meta_info['iter'] = 0
        self.io.record_time()

        loader = self.data_loader['train']
        loss_value = []
        # print(len(loader.dataset))
        for data, label in loader:

            # get data
            data  = data.float().to(self.dev)
            label = label.long().to(self.dev)
            
            label[label==13] = 1
            label[label==90] = 2
            label[label==111] = 3
            label[label==116] = 4

            N,C,T,V,M = data.size()
                     
            # forward
            recon_data, cat_y, latent_z, z = self.model(data)

            # autoencoder loss
            loss = self.loss(recon_data, data, label , cat_y, latent_z)
            
            # backward
            self.optimizer["autoencoder"].zero_grad()
            loss.backward()
            self.optimizer["autoencoder"].step()

            # cat_y discriminator train
            valid = Variable(torch.zeros(label.shape[0], 1 ).fill_(1.0), requires_grad=False).float().to(self.dev)
            fake  = Variable(torch.zeros(label.shape[0], 1 ).fill_(0.0), requires_grad=False).float().to(self.dev)
            
            rand_label = torch.randint(0,self.model.num_class,(1,N)).view(-1)

            one_hot_label = F.one_hot(rand_label, num_classes = self.model.num_class).float().to(self.dev)

            y_loss =  F.binary_cross_entropy(self.model.y_discriminator(one_hot_label.detach()) , valid )
            y_loss += F.binary_cross_entropy(self.model.y_discriminator(cat_y.detach())         , fake  )
            y_loss =  y_loss * 0.5
            
            self.optimizer["y_discriminator"].zero_grad()
            y_loss.backward()
            self.optimizer["y_discriminator"].step()
            
            # latent_z discriminator train
            valid    = Variable(torch.zeros(latent_z.shape[0], 1 ).fill_(1.0), requires_grad=False).float().to(self.dev)
            fake     = Variable(torch.zeros(latent_z.shape[0], 1 ).fill_(0.0), requires_grad=False).float().to(self.dev)

            sample_z = torch.randn_like( latent_z, requires_grad=False )

            z_loss  =  F.binary_cross_entropy(self.model.z_discriminator(sample_z.detach()) , valid )
            z_loss  += F.binary_cross_entropy(self.model.z_discriminator(latent_z.detach()  ) , fake  )
            z_loss  =  z_loss * 0.5

            self.optimizer["z_discriminator"].zero_grad()
            z_loss.backward()
            self.optimizer["z_discriminator"].step()

            #get matched
            (values, indices) = cat_y.max(dim=1)

            # statistics
            self.iter_info['loss']      = loss.data.item()
            # self.iter_info['cat_loss']  = cat_loss.data.item()
            self.iter_info['acc']       = self.get_homogeneity(cat_y,label)
            
            self.iter_info['y_loss']    = y_loss.data.item()
            self.iter_info['z_loss']    = z_loss.data.item()
            self.iter_info['lr']        = '{:.6f}'.format(self.lr)
            self.iter_info['time']      = '{:.6f}'.format(int(time.time() - self.io.cur_time))
            
            loss_value.append( self.iter_info['loss'] )
            
            self.show_iter_info()
            self.meta_info['iter'] += 1
    
        print(indices.view(-1))
        print(label.view( -1 ))
        print((label == indices).sum(),len(label) )
        
        if(not os.path.exists(self.io.work_dir + "/result")):
            os.makedirs(self.io.work_dir + "/result/")

        np.save(self.io.work_dir + "/result/data{}.npy".format(self.meta_info["epoch"]),data.cpu().numpy())
        np.save(self.io.work_dir + "/result/recon{}.npy".format(self.meta_info["epoch"]),recon_data.detach().cpu().numpy())
        
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()

        loader      = self.data_loader['test']
        loss_value  = []
        result_frag = []
        label_frag  = []

        for data, label in loader:
            
            # get data
            data  = data.float().to(self.dev)
            label = label.long().to(self.dev)
            
            label[label==13] = 1
            label[label==90] = 2
            label[label==111] = 3
            label[label==116] = 4

            # evaluation
            with torch.no_grad():
                recon_data, cat_y, latent_z, z = self.model(data)
            
            result_frag.append(cat_y)
            
            # get loss
            if evaluation:

                loss = self.loss( recon_data, data, label, cat_y, latent_z)
                loss_value.append( loss.data.item() )
                label_frag.append( label )

        if(not os.path.exists(self.io.work_dir + "/result")):
            os.makedirs(self.io.work_dir + "/result/")

        np.save(self.io.work_dir + "/result/eval_data{}.npy".format(self.meta_info["epoch"]),data.cpu().numpy())
        np.save(self.io.work_dir + "/result/eval_recon{}.npy".format(self.meta_info["epoch"]),recon_data.detach().cpu().numpy())

        self.result = torch.cat( result_frag )
        
        print(self.result.size())
        if(evaluation):
            self.io.print_log("Evaluation {}:".format(self.meta_info["epoch"]))
            
            self.label                   = torch.cat(label_frag)
            self.epoch_info['label']     = self.label.data.cpu().numpy()
            self.epoch_info['mean_loss'] = np.mean( loss_value )
            self.epoch_info['acc'] = self.get_homogeneity(self.result,self.label)
            self.show_epoch_info()

            # show top-k accuracy
            # for k in self.arg.show_topk:
            #     self.show_topk( k )

    @staticmethod
    def get_parser(add_help = False ):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help = False )
        
        parser = argparse.ArgumentParser(
            add_help    = add_help,
            parents     = [ parent_parser ],
            description = 'Spatial Temporal Graph Convolution Network' 
            )

        # region arguments yapf: disable
        
        # evaluation
        parser.add_argument('--show_topk'   , type=int      , default=[1, 5], nargs='+' , help='which Top K accuracy will be shown')
        
        # optim
        parser.add_argument('--base_lr'     , type=float    , default=0.01              , help='initial learning rate')
        parser.add_argument('--step'        , type=int      , default=[]    , nargs='+' , help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer'                   , default='SGD'             , help='type of optimizer')
        parser.add_argument('--nesterov'    , type=str2bool , default=True              , help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float    , default=0.0001            , help='weight decay for optimizer')
        
        # endregion yapf: enable

        return parser


