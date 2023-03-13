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

def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i]-data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)



def tsne_fit(data, dimension='3D'):
    if dimension == '2D':
        tsne = TSNE(n_components=2, init='pca', random_state=2023)
    elif dimension == '3D':
        tsne = TSNE(n_components=3, init='pca', random_state=2023)
    else:
        raise ValueError('[Error] Wrong Parameter!')
    res = tsne.fit_transform(data)
    return res

   
def plot_embedding(data, label, dimension):

    
    fig = plt.figure()
    ax = plt.subplot(111, projection=dimension)
    color = ['r', 'g', 'b']
    for i in range(data.shape[0]):
        ax.scatter3D(data[i, 0], data[i, 1], data[i, 2],
                     marker='o', color=color[label[i]])
    return fig

if __name__ == '__main__':
    dataset = LandslideDataset(train=True, labeled='unlabeled')
    dataLoader = DataLoader(dataset, batch_size=1000, drop_last=False, shuffle=False)
    for data,_,_,_,f_name in dataLoader:
        images = np.array(data).reshape(-1, 128*128*3)
    
    #--------------------------------------------------------------------------
    #  孤立森林
    # embedding = tsne_fit(images, '3D')

    embedding = np.load('./tsne_3d.npy')
    _max, _min = np.max(embedding), np.min(embedding)
    embedding = (embedding - _min) / (_max - _min)
    iforest = IsolationForest(n_estimators=100, max_samples='auto',  
                        contamination=0.1, max_features=3,  
                        bootstrap=False, n_jobs=-1, random_state=2023)
    
    pred = iforest.fit_predict(embedding)

    
    #-----------------------------------------------------------------------
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(111, projection='3d')
    color = ['r', 'b', 'g']
    for i in range(len(embedding)):
        if pred[i] == 1:
            c = 'b'
        elif pred[i] == -1:
            c = 'r'
        ax.scatter3D(embedding[i, 0], embedding[i, 1], embedding[i, 2], marker='o', s=1, color=c)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    dot1 = ax.scatter3D([], [],[], color='r', s=1)
    # dot2 = ax.scatter3D([], [],[], color='b', s=1) 
    fig.legend(handles=[dot1], labels=['abnormal sample'])
    fig.savefig('./3d_iforest.png')
