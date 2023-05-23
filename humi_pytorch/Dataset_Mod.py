import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot
matplotlib.use('TkAgg')

###Class that provides the data for training
###Requires a folder with numpys, labelled 1-X.npy and another folder with labels using the same naming scheme
#The multi_img variable should always be set to False and is currently not supported
class ImageDataset(Dataset):

    def __init__(self, img_path, mask_path, X, multi_img = False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.multi_img = multi_img

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # img = np.load(self.img_path + self.X[idx] + '.npy')
        if self.multi_img:
            list_img = []
            index=0
            for path in self.img_path:
                img = np.load(path + self.X[idx] + '.npy')
                img = torch.from_numpy(img).float()
                img = img.permute(3, 0, 1, 2)
                index+=1
                list_img.append(img)
            img = list_img

        else:
            img = np.load(self.img_path + self.X[idx] + '.npy')
            img = torch.from_numpy(img).float() #make tensor from numpy
            img = img.permute(3, 0, 1, 2) #alter axis order to fit the Channel*Dim_1*Dim_2*Dim3 standard required by pytorch

        mask = np.load(self.mask_path + self.X[idx] + '.npy')
        mask = torch.from_numpy(mask).float()
        return img, mask

