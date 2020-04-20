#!/usr/bin/env python
# pylint: disable=W0201
import sys
# sys.path.extend(['../'])

import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

from .processor import Processor
import torch.nn.functional as F

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

def display_skeleton(data,sample_name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    data = data.reshape((1,) + data.shape)

    
    print("===========================================")

    # for batch_idx, (data, label) in enumerate(loader):
    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    
    
    ax = fig.add_subplot(111, projection='3d')
   
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    edge = [(1, 2)  ,(2, 21)    ,(3, 21)   ,(4, 3)    ,(5, 21)   ,(6, 5)    , 
            (7, 6)  ,(8, 7)     ,(9, 21)   ,(10, 9)   ,(11, 10)  ,(12, 11)  ,
            (13, 1) ,(14, 13)   ,(15, 14)  ,(16, 15)  ,(17, 1)   ,(18, 17)  ,
            (19, 18),(20, 19)   ,(22, 23)  ,(23, 8)   ,(24, 25)  ,(25, 12)  ]
    edge = [(i-1,j-1) for (i,j) in edge]
    pose = []

    for m in range(M):
        a = []
        for i in range(len(edge)):
            a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            
        pose.append(a)

    ax.axis([-1, 1, -1, 1])
    ax.set_zlim3d(-1, 1)
    if not os.path.exists('/content/image/'+str(sample_name)+"/"):
        os.makedirs('/content/image/'+str(sample_name)+"/")
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[0, :2, t, v1, m]
                x2 = data[0, :2, t, v2, m]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                    pose[m][i].set_3d_properties([data[0, 2, t, v1, m],data[0, 2, t, v2, m]])
                        
        fig.canvas.draw()
        
        
        plt.savefig('/content/image/'+str(sample_name)+"/" + str(t) + '.jpg')
        plt.pause(0.01)

def KM_classifier(data,label,k): 
    
    kmeans = KMeans(n_clusters=k, random_state=0, verbose = 0,max_iter= 10000).fit(data)
    pred = kmeans.labels_

    correct = 0
    for i in range(k):
        temp = label[pred==i]
        value, count = Counter(temp).most_common()[0]
        print("{} -> {} | acc = {}".format(i,value,count/len(temp)))
        correct += count
    print("Total_match : {}".format(correct))
    print("Avg accuracy : {}".format(correct/len(label)))

    return correct/len(label)

def loss_function(recon_x, x, mu, logvar):
    n = recon_x.size(0)
    # print(recon_x.shape)
    # BCE = F.binary_cross_entropy(recon_x.view(n,-1), x.view(n,-1), reduction='sum')
    BCE = nn.functional.mse_loss(recon_x, x)
    # print("BCE : ", BCE)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # print(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1).mean())
    # print(mu.pow(2).shape)
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1).mean()
    return BCE + KLD

class Cluster(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
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
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))


    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        i = 0
        for data, label in loader:
            
            i += 1
            if((i % 100 )== 0):
                self.io.print_log('Iteraction : {}'.format(i))
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                recon_data, mean, lsig, z = self.model(data)

            result_frag.append(mean.data.cpu().numpy())
            
            # display_skeleton(data[2].cpu().numpy())
            

            # get loss
            if evaluation:
                loss = loss_function(recon_data,data,mean,lsig)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
        
        display_skeleton(data[2].cpu().numpy(),"ori")
        display_skeleton(recon_data[2].cpu().numpy(),"recon")

        self.io.print_log("Start KMeans")
        # accuracy_pred = KM_classifier(np.concatenate(result_frag),np.concatenate(label_frag),120)

        self.result = accuracy_pred

        if evaluation:
            # self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            # for k in self.arg.show_topk:
            #     self.show_topk(k)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # test phase
        

        # the path of weights must be appointed
        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.arg.model))
        self.io.print_log('Weights: {}.'.format(self.arg.weights))

        # evaluation
        self.io.print_log('Evaluation Start:')
        self.test()
        self.io.print_log('Done.\n')

        # save the output of model
        if self.arg.save_result:
            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'test_result.pkl')
    
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


