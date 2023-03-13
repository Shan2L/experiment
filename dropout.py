import torch 
import numpy as np
from nets.deeplabv3_plus import DeepLab
from torchvision import transforms
from data_loader import LandslideDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os 
import scipy.misc
from tqdm import tqdm
from utils.utils_metrics import f_score, compute_auc

def MK_Dropout(model_path, batch_data):
    dropout = torch.nn.Dropout(0.3)
    dropout.train()
    
    
    model = DeepLab(num_classes=2, backbone='mobilenet', downsample_factor=8, pretrained=True)
    model.load_state_dict(torch.load(model_path))
    
  
    feature = model.backbone.features[0][0](batch_data)
    mid_f0 = dropout(feature)
    
    mid_f1 = model.backbone.features[0][1:](mid_f0)
    low_level_feature = model.backbone.features[1:4](mid_f1)
    mid_f2 = model.backbone.features[4:](low_level_feature)
    
    mid_f3 = model.aspp(mid_f2)
    low_level_features = model.shortcut_conv(low_level_feature)
    mid_f4 = F.interpolate(mid_f3, size=(low_level_features.size(2), low_level_feature.size(3)), mode='bilinear', align_corners=True)
    res = model.cat_conv(torch.cat((mid_f4, low_level_features), dim=1))
    res = model.cls_conv(res)
    res = F.interpolate(res, size=(test.size(2), test.size(3)), mode='bilinear', align_corners=True)
    res = res.transpose(1,2).transpose(2,3)
    soft_max = torch.softmax(res, axis=-1)
    result = soft_max.argmax(axis=-1).detach().numpy()
    
    print(result.shape)  
    return result  

if __name__  == '__main__':
    model_path = '/exp/deeplabv3+/logs/ep100-loss0.021-val_loss0.046.pth'
    test  = torch.from_numpy(np.zeros((128, 3, 128, 128))).float()
    # mask_path = './logs/loss_2023_03_08_12_50_34/mask/'
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = LandslideDataset(train=True, labeled='unlabeled', transform=train_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=False)
    temp_res = []
    for image, mask, label, _, file_paths in data_loader:
        for i in range(10):
            uncertein_map = MK_Dropout(model_path, batch_data=image)
            print(uncertein_map.shape)
            temp = uncertein_map[3].reshape(128, 128)*255
            print(temp.shape)
            scipy.misc.imsave('./{}.png'.format(i), temp)
            temp_res.append(temp)
        break
    temp_res  = np.array(temp_res)
    temp_res  = np.logical_xor(temp_res[0], temp_res[1])
    print(temp_res)
         
                      

    
    
    

    

    
    # pred = res.transpose(1, 2).transpose(2, 3).detach().numpy()
    # pred = pred.argmax(axis=-1)
    
    
    # for mask, f_name in tqdm(zip(pred, path_lists)):
    #     file_name = f_name.split('/')[-1].replace('image', 'mask')
    #     save_path = os.path.join(mask_path, file_name)
    #     print(save_path)
    #     scipy.misc.imsave(save_path, mask*255)