from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, MaxPooling2D
from skimage.measure import shannon_entropy
import  matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import h5py
import scipy
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
def mean_segment(image, segments ,single_channel=True, save_seg=False):
    if single_channel:
        if save_seg:
            out = mark_boundaries(image,segments)
            print('[INFO] Saving segment png')
            plt.imshow(out)
            plt.savefig('segment.png')

        num_seg = np.max(segments)
        for index in range(1, num_seg + 1):
            val = image[segments == index]
            mean = np.mean(val)
            image[segments == index] = mean

        return image

    else:

        r, g, b = cv2.split(image)
        num_seg = np.max(segments)

        for index in range(1, num_seg+1):
            val_r = r[segments == index]
            mean_r = np.mean(val_r)
            r[segments == index] = mean_r

            val_g = g[segments == index]
            mean_g = np.mean(val_g)
            g[segments == index] = mean_g

            val_b = b[segments == index]
            mean_b = np.mean(val_b)
            b[segments == index] = mean_b

        result = cv2.merge([r, g, b])
        return result

def center_crop(im, output_size):
    output_height, output_width = output_size
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("[Exception] The size of image is too small to crop")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]


def mean_slic(data, n_segments, compactness, sigma, gray_scale=True, shape=None):

    segments = slic(data, n_segments=n_segments, compactness=compactness)
    res_img = mean_segment(image=data, segments=segments, single_channel=gray_scale, save_seg=False)
    # scipy.misc.imsave('./eval/celeba/slic/{}.png'.format(i), res_img)
    mean_img_array = np.array(res_img, dtype=np.float32).reshape(shape)


    return mean_img_array

def compute_shannon_entropy(images):
    total_r = 0
    total_g = 0
    total_b = 0
    for image in images:
        enc_r = shannon_entropy(image[:, :, 0])
        enc_g = shannon_entropy(image[:, :, 1])
        enc_b = shannon_entropy(image[:, :, 2])
        total_r += enc_r
        total_g += enc_g
        total_b += enc_b
    mean_r = total_r / len(images)
    mean_g = total_g / len(images)
    mean_b = total_b / len(images)
    return mean_r, mean_g, mean_b

def compute_ssim(images1, images2):
    total = 0
    for img1, img2 in zip(images1, images2):
        _ssim = ssim(img1, img2, multichannel=True)
        total += _ssim
    return total/ len(images1)
            
            

pattern = '/exp/deeplabv3+/dataset/visual/image/*.png'
file_list = glob.glob(pattern)
file_list.sort()

raw_images = []
slic_result = []
input = Input(shape=(128, 128, 3))
output = MaxPooling2D(pool_size=(5, 5), strides=(2,2), padding='same')(input)
model1 = Model(input, output)

input = Input(shape=(64, 64, 3))
output = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same')(input)
model2 = Model(input, output)

for f in tqdm(file_list[1000:1010]):
    name = f.split('/')[-1]
    img = scipy.misc.imread(f, mode='RGB')
    scipy.misc.imsave('./slic/oringin_{}'.format(name), img)
    raw_images.append(img)
    
    down_sampled = model1.predict(img.reshape(1, 128, 128, 3))
    img = mean_slic(down_sampled.reshape(64, 64, 3), n_segments=500, compactness=10, sigma=0, gray_scale=False, shape=(64, 64, 3))
    scipy.misc.imsave('./slic/1_{}'.format(name), img)
    print(img.shape)
    down_sampled2 = model2.predict(img.reshape(1, 64, 64, 3))
    print(down_sampled2.shape)
    img = mean_slic(down_sampled2.reshape(32, 32, 3), n_segments=100, compactness=10, sigma=0, gray_scale=False, shape=(32, 32, 3))
    scipy.misc.imsave('./slic/2_{}'.format(name), img)
    
    slic_result.append(img)


# before_shannon = compute_shannon_entropy(raw_images)
# after_shannon = compute_shannon_entropy(slic_result)

# print(before_shannon[0], before_shannon[1], before_shannon[2])
# print(after_shannon[0], after_shannon[1], after_shannon[2])

# ssim = compute_ssim(raw_images, slic_result)
# print('ssim:{}'.format(ssim))
    
    






