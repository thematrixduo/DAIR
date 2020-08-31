from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch.utils.data
from sklearn.utils import shuffle

# Change figure aesthetics
#sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 1.5})
'''
class DspritesDataset(Dataset):
    """dsprites dataset."""

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
'''



# Load dataset
def load_data(data_path,batch_size):
    train_file_path = os.path.join(data_path,'dsprites_data.npz')
    train_dataset_zip = np.load(train_file_path)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_imgs = train_dataset_zip['images']
    train_num_obj = train_dataset_zip['labels']

    #test_imgs = test_dataset_zip['images']
    #test_labels = test_dataset_zip['labels']
    #test_digits = test_dataset_zip['digits']
    print(train_imgs.shape)

    train_imgs = np.reshape(train_imgs,(train_imgs.shape[0],1,train_imgs.shape[1],train_imgs.shape[2]))
    #test_imgs = np.reshape(test_imgs,(test_imgs.shape[0],1,test_imgs.shape[1],test_imgs.shape[2]))

    train_imgs,train_num_obj = shuffle(train_imgs,train_num_obj)
    print('data reshuffled')
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_imgs).float(),torch.from_numpy(train_num_obj).float())
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, drop_last=True,**kwargs)
    #test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_imgs).float(),torch.from_numpy(test_labels).float())
    #test_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, drop_last=True,**kwargs)
    #metadata = dataset_zip['metadata'][()]
    return train_loader, torch.from_numpy(train_imgs).float(),torch.from_numpy(train_num_obj).float()#, test_loader

def load_data_label(data_path,batch_size,dataset):
    train_file_path = os.path.join(data_path,dataset)
    train_dataset_zip = np.load(train_file_path)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_imgs = train_dataset_zip['images']
    train_imgs = train_imgs/255
    print(np.min(train_imgs),np.max(train_imgs))
    train_labels = train_dataset_zip['labels']
    train_num_obj = np.sum(np.clip(train_labels+1,0,1),-1).astype(np.int)
    train_labels = np.concatenate([train_labels,-np.ones((train_labels.shape[0],1))],-1)
    labels_mask = np.clip(train_labels,-1,0)
    
    #test_imgs = test_dataset_zip['images']
    #test_labels = test_dataset_zip['labels']
    #test_digits = test_dataset_zip['digits']
    print(train_imgs.shape)

    train_imgs = np.reshape(train_imgs,(train_imgs.shape[0],1,train_imgs.shape[1],train_imgs.shape[2]))
    #test_imgs = np.reshape(test_imgs,(test_imgs.shape[0],1,test_imgs.shape[1],test_imgs.shape[2]))

    train_imgs, train_num_obj, train_labels, labels_mask = shuffle(train_imgs,train_num_obj,train_labels, labels_mask)
    print('data reshuffled')
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_imgs).float(),torch.from_numpy(train_num_obj).float())
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, drop_last=True,**kwargs)
    #test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_imgs).float(),torch.from_numpy(test_labels).float())
    #test_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, drop_last=True,**kwargs)
    #metadata = dataset_zip['metadata'][()]
    return train_loader, torch.from_numpy(train_imgs).float(),torch.from_numpy(train_num_obj).float(), train_labels, labels_mask


