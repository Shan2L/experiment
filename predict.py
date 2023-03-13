import torch 
import numpy as np
from nets.deeplabv3_plus import DeepLab
from torchvision import transforms
from data_loader import LandslideDataset
from torch.utils.data import DataLoader
import os 
import scipy.misc
from tqdm import tqdm
from utils.utils_metrics import f_score, compute_auc

if __name__  == '__main__':
    
    model_path = '/exp/deeplabv3+/logs/ep100-loss0.021-val_loss0.046.pth'
    model = DeepLab(num_classes=2, backbone='mobilenet', downsample_factor=8, pretrained=True)
    model.load_state_dict(torch.load(model_path))
    
    mask_path = './logs/loss_2023_03_08_12_50_34/mask/'
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = LandslideDataset(train=True, labeled='unlabeled', transform=train_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False, drop_last=False)
    
    for image, mask, label, _, file_paths in data_loader:
        res = model(image)
        path_lists = file_paths 
        _f_score = f_score(res, label)
        auc = compute_auc(res, mask.view((-1, 1, 128 ,128)))
        print(_f_score)
        print(auc)

        break
    
    pred = res.transpose(1, 2).transpose(2, 3).detach().numpy()
    pred = pred.argmax(axis=-1)
    
    
    for mask, f_name in tqdm(zip(pred, path_lists)):
        file_name = f_name.split('/')[-1].replace('image', 'mask')
        save_path = os.path.join(mask_path, file_name)
        print(save_path)
        scipy.misc.imsave(save_path, mask*255)