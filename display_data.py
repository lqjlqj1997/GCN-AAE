import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def display_skeleton(data,sample_name,save=False):
    
    data = data.reshape((1,) + data.shape)

    N, C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    
    ax = fig.add_subplot(111, projection='3d')
   
    p_type = ['b-', 'g-', 'r-', 'c-', ' m-', 'y-', 'k-', 'k-', 'k-', 'k-']
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
    ax.view_init(elev=15, azim=45)

    if (save is True):
        if not os.path.exists('./image/'+str(sample_name)+"/"):
            os.makedirs('./image/'+str(sample_name)+"/")
    
    for t in range(T):
        for m in range(M):

            for i, (v1, v2) in enumerate(edge):
                x1 = data[0, :2, t, v1, m]#.around(decimals=2)
                x2 = data[0, :2, t, v2, m]
                
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1 :
                    pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                    pose[m][i].set_3d_properties([data[0, 2, t, v1, m],data[0, 2, t, v2, m]])    
                
                    
        fig.suptitle('T = {}'.format(t), fontsize=16)            
        fig.canvas.draw()
        
        
        if (save is True):
            plt.savefig('./image/'+str(sample_name)+"/" + str(t) + '.jpg')

        plt.pause(1/240)
    plt.close(fig)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( add_help=add_help, description='Display Skeleton')

    parser.add_argument('--data' ,default= ""    , help='path with the orignal data')
    parser.add_argument('--recon',default= ""    , help='path of reconstructed data')
    parser.add_argument('--save' ,type= str2bool ,default= False , help='save Figure of skeleton')
    
    arg = parser.parse_args()
    
    data_disp  = False
    recon_disp = False
    
    if(os.path.exists(arg.data)) :
        data      = np.load(arg.data)
        data_disp = True
    else:
        print("Invalid or No Path of data is provided")

    if(os.path.exists(arg.recon)) :
        recon      = np.load(arg.recon)
        recon_disp = True
    else:
        print("Invalid or No Path of reconstructed data is provided")

    data = np.load("./data0.npy")
    recon_data = np.load("./recon0.npy")
 
    for i in range(32):
        print("====== data {} =======".format(i))
        
        if(data_disp):
            display_skeleton(data[i]      ,"original_{}".format(i), save= arg.save)
        
        if(recon_disp):
            display_skeleton(recon_data[i],"recon_{}".format(i), save= arg.save)

