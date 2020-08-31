import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from air_model_dis_deep_test import DrawModel
from config_multi_dsprites import *
from utility import Variable,save_image,xrecons_grid
import torch.nn.utils
import matplotlib.pyplot as plt
from functools import partial
import visdom
import itertools
from load_data import load_data_label 
from observations import multi_mnist
from viz import draw_many, tensor_to_objs, latents_to_tensor

cuda = False
data_path = '/local/scratch/wd263/dsprites-dataset/'
model_save_path = 'save/weights_210000.tar'
torch.set_default_tensor_type('torch.FloatTensor')

device = torch.device("cuda" if cuda else "cpu")


#train_loader, X, true_counts, labels, labels_mask = load_data_label(data_path,batch_size,'multi_dsprites_lb_large_G0d6.npz')
print('open Visdom')
vis = visdom.Visdom(env='AIR_dspritesTest')


T=5
model = DrawModel(T,A,B,z_size,cat_size,window_size,rnn_size,device=device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if cuda:
    model.cuda()


def test():
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    generate_image(666)

def rotation_matrix(angle,batch_size):
    mat = np.zeros((batch_size,6))
    mat[:,0] = np.cos(angle)
    mat[:,1] = np.sin(angle)
    mat[:,3] = -np.sin(angle)
    mat[:,4] = np.cos(angle)
    return mat

def generate_image(count,gen_batch_size=16):

    obj_list = []
    z_where_list = []
    rot_mat_list = []
    for i in range(T-1):
        
        obj = np.random.randint(3)
        obj_tensor = torch.zeros(gen_batch_size,cat_size).scatter_(1,torch.tensor(obj).expand(gen_batch_size,1),1).float().to(device)
        obj_list.append(obj_tensor)

        scale = np.random.uniform(1.8,2.3,size=(gen_batch_size,1))
        pos = np.random.uniform(-1.7,1.7,size=(gen_batch_size,2))
        angle = np.random.uniform(-3.14,3.14,size=(gen_batch_size))
        z_where = torch.from_numpy(np.concatenate([scale,pos],axis=-1)).float().to(device)
        z_where_list.append(z_where)
        rot_mat = rotation_matrix(angle,gen_batch_size)
        rot_mat_list.append(rot_mat)
        print('object num{}, obj: {}, scale:{}, pos:{}'.format(i,obj,scale[0], pos[0]))
        
    gen_x = model.generate(obj_list,z_where_list,rot_mat_list,batch_size = gen_batch_size)
    save_image(gen_x,count,'gen',path='X_image/')


if __name__ == '__main__':

    test()
