from nets.deeplabv3_plus import DeepLab
import numpy as np
import torch

model   = DeepLab(num_classes=2, backbone='mobilenet', downsample_factor=8, pretrained=True)

input_ = np.random.randint(low=0, high=1, size=(128, 3, 128, 128))
input_ = torch.from_numpy(input_).float()

output_ = model(input_)
print(output_.shape)
