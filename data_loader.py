# import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
import numpy as np
import os, glob
import json
import scipy.misc
import h5py


class LandslideDataset(Dataset):
    def __init__(self, train=True, labeled='labeled', transform=None):
        self.train = train
        self.labeled = labeled
        self.transform = transform
        self.image_dir = './dataset/visual/image'
        self.mask_dir = './dataset/visual/mask'
        self.train_split = './dataset/split/train.json'
        self.test_split = './dataset/split/test.json'
        if self.train:
            with open(self.train_split, 'r') as f:
                data_dic = json.load(f)
                if self.labeled == 'labeled':
                    self.data_list = data_dic['labeled']
                elif self.labeled == 'unlabeled':
                    self.data_list = data_dic['unlabeled']
                else:
                    raise ValueError('[Error] Wrong parameter!')
        else:
            with open(self.test_split, 'r') as f:
                data_dic = json.load(f)
                self.data_list = data_dic

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_file = os.path.join(self.image_dir, self.data_list[index]['img'])
        mask_file = os.path.join(self.mask_dir, self.data_list[index]['mask'])
        image = scipy.misc.imread(image_file)
        mask = scipy.misc.imread(mask_file)
        mask = mask // 255.0
        mask = mask.astype(np.int)
        label = self.data_list[index]['label']

        seg_labels = np.eye(2)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape(128, 128, 2)

        if self.transform is not None:
            image = self.transform(image).float()  
            seg_labels = self.transform(seg_labels).float()  
        label = torch.from_numpy(np.array(int(label))).float()
        mask = torch.from_numpy(mask).float()

        return image, mask, seg_labels, label

def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = LandslideDataset(train=False, labeled='unlabeled', transform=train_transform)
    img, mask, seg_labels, label = dataset[0]
    gen             = DataLoader(dataset, shuffle = True, batch_size = 128, num_workers = 0, pin_memory=False,
                                drop_last = False)
    for item in gen:
        print(item[0].shape)
        print(item[1].shape)
        print(item[2].shape)
    print(len(gen))

