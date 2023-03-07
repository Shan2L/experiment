import numpy as np
import glob
import os 
import h5py
import scipy.misc
import json 
from tqdm import tqdm
import random


np.random.seed(2023)
random.seed(2023)
img_dir = './h5_file/img'
mask_dir = './h5_file/mask' 
visual_img_dir = './visual/image/'
visual_mask_dir = './visual/mask/'

file_pattern = os.path.join(img_dir, '*.h5')
file_list = sorted(glob.glob(file_pattern))

dic = {}
landslide_array = []
non_landslide_array = []


for img_file in file_list:
    index = img_file.split('/')[-1].split('.')[0].split('_')[-1]
    mask_file = img_file.replace('img', 'mask').replace('image', 'mask')
    img_name = img_file.split('/')[-1]
    mask_name = mask_file.split('/')[-1]
    with h5py.File(mask_file, 'r') as f, h5py.File(img_file, 'r') as imf:
        img = imf['img'][:]
        img = np.array(img)

        mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

        for i in range(len(mean)):
            img[:, :, i] -= mean[i]
            img[:, :, i] /= std[i]

        img = img[:, :, 3:0:-1]

        mask = f['mask']
        mask = np.array(mask)
        img_file_name = img_file.split('/')[-1]
        mask_file_name = mask_file.split('/')[-1]
        index = img_file_name.split('.')[0].split('_')[-1]


        if mask[mask==1].size == 0:
            non_landslide_array.append({'img_name':img_file_name.replace('h5', 'png') ,'image':img, 'mask_name':mask_file_name.replace('h5', 'png') ,'mask':mask})
        else:
            landslide_array.append({'img_name':img_file_name.replace('h5', 'png') ,'image':img, 'mask_name':mask_file_name.replace('h5', 'png') ,'mask':mask})

landslide_info = []
non_landslide_info = []

print('[Info]Saving landslide data... ')
for item in tqdm(landslide_array):
    scipy.misc.imsave(os.path.join(visual_img_dir, item['img_name']), item['image'])
    scipy.misc.imsave(os.path.join(visual_mask_dir, item['mask_name']), item['mask']*255)
    landslide_info.append({'img':item['img_name'], 'mask':item['mask_name'], 'label':'1'})

print('[Info]Saving non_landslide data... ')
for item in tqdm(non_landslide_array):
    scipy.misc.imsave(os.path.join(visual_img_dir, item['img_name']), item['image'])
    scipy.misc.imsave(os.path.join(visual_mask_dir, item['mask_name']), item['mask']*255)
    non_landslide_info.append({'img':item['img_name'], 'mask':item['mask_name'], 'label':'0'})

print('The number of landslide image is {}'.format(len(landslide_info)))
print('The number of non_landslide image is {}'.format(len(non_landslide_info)))

with open('./split/landslide.json', 'w') as lsf, open('./split/non_landslide.json', 'w') as nlsf:
    json.dump(landslide_info, lsf, indent='\n')
    json.dump(non_landslide_info, nlsf, indent='\n')















