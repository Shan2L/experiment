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
    
    # embedding = tsne_fit(images, '3D')
    embedding = np.load('./tsne_3d.npy')
    _max, _min = np.max(embedding), np.min(embedding)
    embedding = (embedding - _min) / (_max - _min)
    
    k = 7
    k_dist = select_MinPts(embedding, k)
    k_dist.sort()
    plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
    plt.savefig('./anomaly/k_dist.png')
    plt.close('ALL')
    eps = k_dist[::-1][70]
    plt.scatter(70,eps,color="r")
    plt.plot([0,70],[eps,eps],linestyle="--",color = "r")
    plt.plot([70,70],[0,eps],linestyle="--",color = "r")
    plt.ylim(0, 0.5)
    plt.savefig('./eps_value')
    plt.close('ALL')

    
    
    labels = DBSCAN(eps=eps, min_samples=k+1).fit_predict(embedding)
    print(np.unique(labels).shape)
    print(labels)
    
    
    
    mean_vector = [np.mean(embedding[0]), np.mean(embedding[1]), np.mean(embedding[2])]

    distance = []
    for item in embedding:
        d = (item[0]-mean_vector[0])**2+(item[1]-mean_vector[1])**2+(item[2]-mean_vector[2])**2
        d = np.sqrt(d)
        distance.append(d)
        
    distance = np.array(distance)
    distance = distance[np.where(labels!=-1)[0]]
    flierprops = dict(marker='_', markeredgecolor='red')
    plt.figure(figsize=(3, 4))
    fig = sns.boxplot(data=distance, width=0.01, linewidth=1, flierprops=flierprops)
    fig.set_xlim(-0.1,0.1)
    fig.set_ylim(0,1)
    # sns.swarmplot(data=distance, color="grey")
    fig = fig.get_figure()
    fig.savefig('./box_plot_dbscan.png')
    
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(111, projection='3d')
    color = ['r', 'b', 'g']
    for i in range(len(embedding)):
        if labels[i] == 1:
            c = 'k'
        elif labels[i] == -1:
            c = 'r'
        elif labels[i] == 0:
            c = 'b'
        elif labels[i] == 2:
            c = 'y'
        ax.scatter3D(embedding[i, 0], embedding[i, 1], embedding[i, 2], marker='o', s=1, color=c)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    dot1 = ax.scatter3D([], [],[], color='r', s=1)
    # plt.tick_params(labelsize=8)
    # dot2 = ax.scatter3D([], [],[], color='k', s=1) 
    # dot3 = ax.scatter3D([], [],[], color='b', s=1) 
    # dot4 = ax.scatter3D([], [],[], color='y', s=1) 
    fig.legend(handles=[dot1], labels=['abnormal sample'])
    fig.savefig('./3d_dbscan.png')

    print(np.where(labels==-1)[0].shape)
