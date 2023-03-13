import numpy as np
from time import time
import cv2
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from data_loader import LandslideDataset
from torch.utils.data import DataLoader
import  scipy.misc 
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import seaborn as sns



embedding = np.load('./tsne_3d.npy')
_max, _min = np.max(embedding), np.min(embedding)
embedding = (embedding - _min) / (_max - _min)
iforest = IsolationForest(n_estimators=100, max_samples='auto',  
                    contamination=0.1, max_features=3,  
                    bootstrap=False, n_jobs=-1, random_state=2023)

pred = iforest.fit_predict(embedding)

embedding = embedding[np.where(pred!=-1)[0]]
labels = [0 for i in range(len(embedding))]


index = np.random.randint(low=0, high=len(embedding), size=200)
repeated = embedding[index]
repeated = np.concatenate([repeated,repeated[:100]], axis=0)
repeated = np.concatenate([repeated,repeated[:100]], axis=0)
repeated = np.concatenate([repeated,repeated[:100]], axis=0)
repeated = np.concatenate([repeated,repeated[:100]], axis=0)
generated = []
for i in range(500):
    sample = []
    x = repeated[i][0] + np.random.normal(loc=0, scale=0.01)
    y = repeated[i][1] + np.random.normal(loc=0, scale=0.01)
    z = repeated[i][2] + np.random.normal(loc=0, scale=0.01)
    sample.append(x)
    sample.append(y)
    sample.append(z)
    sample = np.array(sample)
    generated.append(sample)
generated = np.array(generated)

labels2 = [1 for i in range(500)]
labels.extend(labels2)
embedding = np.concatenate([embedding, generated], axis=0)
print(embedding.shape)
print(len(labels))

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111, projection='3d')
color = ['r', 'b', 'g']
for i in range(len(embedding)):
    if labels[i] == 0:
        c = 'b'
    elif labels[i] == 1:
        c = 'y'
    ax.scatter3D(embedding[i, 0], embedding[i, 1], embedding[i, 2], marker='o', s=5, color=c)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.zaxis.set_major_locator(plt.MultipleLocator(0.1))
dot1 = ax.scatter3D([], [],[], color='b', s=3)
dot2 = ax.scatter3D([], [],[], color='y', s=3) 
fig.legend(handles=[dot1, dot2], labels=['origin', 'generated'])
fig.savefig('./3d_generated.png')