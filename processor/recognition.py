#!/usr/bin/env python
# pylint: disable=W0201
import sys
# sys.path.extend(['../'])

import argparse
import yaml
import numpy as np
import time

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


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

def between_frame_loss(gait1, gait2,args):
    
    N,C,T,V,M = gait1.size()

    g1 = gait1.permute(0, 2, 3, 1, 4).contiguous().view(gait1.shape[0], gait1.shape[2], gait1.shape[1]*gait1.shape[3])
    g2 = gait2.permute(0, 2, 3, 1, 4).contiguous().view(gait2.shape[0], gait2.shape[2], gait2.shape[1]*gait2.shape[3])
    
    num_batches = g1.shape[0]
    num_tsteps = g2.shape[1]
    
    # mid_tstep = np.int(num_tsteps / 2) - 1

    loss = nn.functional.mse_loss(g1, g2,**args)
    

    #motion_loss
    t1 = g1[:,2:] - 2 * g1[:,1:-1] + g1[:,:-2]
    t2 = g2[:,2:] - 2 * g2[:,1:-1] + g2[:,:-2]

    loss += nn.functional.mse_loss(t1,t2,**args)
        
    # loss += nn.functional.mse_loss(g1[:, tidx, :]-g1[:, 0, :]         , g2[:, tidx, :]-g2[:, 0, :], reduction = "sum")
    # loss += nn.functional.mse_loss(g1[:, tidx, :]-g1[:, mid_tstep, :] , g2[:, tidx, :]-g2[:, mid_tstep, :] ,reduction = "sum")
    # loss += nn.functional.mse_loss(g1[:, tidx, :]-g1[:, -1, :]        , g2[:, tidx, :]-g2[:, -1, :], reduction = "sum")        

    return loss         




class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    

    def loss(self,recon_x, x, mu, logvar):
        # args = { "size_average":False,"reduce": True, "reduction" : "sum"}
        args = {"reduction" : "mean"}
        N,C,T,V,M = x.size()
        valid = Variable(torch.zeros(x.shape[0], 1 ).fill_(1.0), requires_grad=False).float().to(self.dev)
        
        BCE = 0
        for m in range(M):
            BCE += between_frame_loss(recon_x[:,:,:,:,m].view(N,C,T,V,1), x[:,:,:,:,m].view(N,C,T,V,1),args)

        KLD =  F.binary_cross_entropy(self.model.y_discriminator(mu), valid, **args )
        KLD += F.binary_cross_entropy(self.model.z_discriminator(logvar), valid,**args )

        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1).mean()
        
        return 0.998 * BCE + 0.002 * KLD

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

    
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.step:
            lr = self.arg.base_lr * (0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
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
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            recon_data, mean, logvar, z = self.model(data)

            loss = self.loss(recon_data, data, mean, logvar)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # self.model.decoder.zero_grad()
            # self.optimizer.step()

            self.model.y_discriminator.zero_grad()
            self.model.z_discriminator.zero_grad()

            #discriminator train
            valid = Variable(torch.zeros(label.shape[0], 1 ).fill_(1.0), requires_grad=False).float().to(self.dev)
            fake  = Variable(torch.zeros(label.shape[0], 1), requires_grad=False).float().to(self.dev)
            
            label       = F.one_hot(label, num_classes = 120).float().to(self.dev)
            sample_z    = torch.randn_like(logvar)

            # self.optimizer.zero_grad()
            y_loss =  F.binary_cross_entropy(self.model.y_discriminator(label.detach()), valid )
            y_loss += F.binary_cross_entropy(self.model.y_discriminator(mean.detach()), fake )
            y_loss = y_loss * 0.5
            y_loss.backward()

            z_loss = F.binary_cross_entropy(self.model.z_discriminator(sample_z.detach()),valid )
            z_loss += F.binary_cross_entropy(self.model.z_discriminator(logvar.detach()), fake )
            z_loss = z_loss * 0.5
            z_loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['y_loss'] = y_loss.data.item()
            self.iter_info['z_loss'] = z_loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['time'] = '{:.6f}'.format(int(time.time() - self.io.cur_time))
            
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        np.save("/content/result/data{}.npy".format(self.meta_info["epoch"]),data.cpu().numpy())
        np.save("/content/result/recon{}.npy".format(self.meta_info["epoch"]),recon_data.detach().cpu().numpy())
        
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            
            # inference
            with torch.no_grad():
                recon_data, mean, lsig, z = self.model(data)
            
            result_frag.append(mean.data.cpu().numpy())
            
            # get loss
            if evaluation:
                loss = self.loss(recon_data,data,mean,lsig)
                loss_value.append(loss.item())
                
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.io.print_log("Evaluation {}:".format(self.meta_info["epoch"]))
            self.epoch_info['label'] = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser


