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

    mean_vector = [np.mean(embedding[0]), np.mean(embedding[1]), np.mean(embedding[2])]

    distance = []
    for item in embedding:
        d = (item[0]-mean_vector[0])**2+(item[1]-mean_vector[1])**2+(item[2]-mean_vector[2])**2
        d = np.sqrt(d)
        distance.append(d)
        
    distance = np.array(distance)
    distance = distance[np.where(pred!=-1)[0]]
    flierprops = dict(marker='_', markeredgecolor='red')
    plt.figure(figsize=(3, 4))
    fig = sns.boxplot(data=distance, width=0.01, linewidth=1, flierprops=flierprops)
    # fig = sns.swarmplot(data=distance, color="grey", )
    fig.plot()
    fig.set_xlim(-0.1,0.1)
    fig.set_ylim(0,1)
    fig = fig.get_figure()
    fig.savefig('./box_plot_iforest.png')
    
    

    print(np.where(pred==-1)[0].shape)
    # for index in np.where(distance>0.3)[0]: