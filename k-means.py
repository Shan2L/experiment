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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score

def draw_elbow(data):
    n_cluster = range(1, 10)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    scores = [kmeans[i].inertia_ for i in range(len(kmeans))]
    fig, ax = plt.subplots()
    ax.plot(n_cluster, scores)
    plt.savefig('./elblow.png')
    
def draw_silhouette(data):
    n_cluster = range(2, 10)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    pred = [kmeans[i].predict(data) for i in range(len(kmeans))]
    scores = [silhouette_score(data, pred[i]) for i in range(len(kmeans))]
    fig, ax = plt.subplots()
    
    ax.plot(n_cluster, scores)
    plt.savefig('./silhouette.png')
    
if __name__ == '__main__':
    
    dataset = LandslideDataset(train=True, labeled='unlabeled')
    dataLoader = DataLoader(dataset, batch_size=1000, drop_last=False, shuffle=False)
    for data,_,_,_,f_name in dataLoader:
        images = np.array(data).reshape(-1, 128*128*3)
    
    # embedding = tsne_fit(images, '3D')
    embedding = np.load('./tsne_3d.npy')
    _max, _min = np.max(embedding), np.min(embedding)
    embedding = (embedding - _min) / (_max - _min)

    
    
    # draw_elbow(embedding)
    # draw_silhouette(embedding)
    distances = []
    means = KMeans(n_clusters=3, random_state=23).fit(embedding)
    pred = means.predict(embedding)
    print(pred)
    for i in range(len(embedding)):
        if pred[i] == 0:
           distance = (embedding[i][0]-means.cluster_centers_[0][0])**2 + (embedding[i][1]-means.cluster_centers_[0][1])**2 +(embedding[i][2]-means.cluster_centers_[0][2])**2 
           distance = distance **2
        elif pred[i] == 1:
           distance = (embedding[i][0]-means.cluster_centers_[1][0])**2 + (embedding[i][1]-means.cluster_centers_[1][1])**2 +(embedding[i][2]-means.cluster_centers_[1][2])**2 
           distance = distance **2
        elif pred[i] == 2:
           distance = (embedding[i][0]-means.cluster_centers_[2][0])**2 + (embedding[i][1]-means.cluster_centers_[2][1])**2 +(embedding[i][2]-means.cluster_centers_[2][2])**2 
           distance = distance **2
        distances.append(distance)

    print(distances)
    distances = np.array(distances)
    
    labels = [-1 if distances[i] > 0.0002 else pred[i] for i in range(0, len(embedding))]
    labels = np.array(labels)

    print(np.where(labels==-1)[0].shape)
    # print(labels)    
    
    mean_vector = [np.mean(embedding[0]), np.mean(embedding[1]), np.mean(embedding[2])]

    distance = []
    for item in embedding:
        d = (item[0]-mean_vector[0])**2+(item[1]-mean_vector[1])**2+(item[2]-mean_vector[2])**2
        d = np.sqrt(d)
        distance.append(d)
        
    distance = np.array(distance)
    distance = distance[np.where(labels!=-1)[0]]
    # print(distance.shape)

    
    flierprops = dict(marker='_', markeredgecolor='red')
    plt.figure(figsize=(3, 4))
    fig = sns.boxplot(data=distance, width=0.01, linewidth=1, flierprops=flierprops)
    fig.set_xlim(-0.1,0.1)
    fig.set_ylim(0, 1)
    # sns.swarmplot(data=distance, color="grey")
    fig = fig.get_figure() 
    fig.savefig('./box_plot_kmeans.png')

    # print(np.where(labels==1)[0].shape)
    
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(111, projection='3d')
    color = ['r', 'b', 'g']
    for i in range(len(embedding)):
        if labels[i] == 1:
            c = 'c'
        elif labels[i] == -1:
            c = 'r'
        elif labels[i] == 0:
            c = 'b'
        elif labels[i] == 2:
            c = 'y'
        ax.scatter3D(embedding[i, 0], embedding[i, 1], embedding[i, 2], marker='o', s=1, color=c)
    ax.scatter3D(means.cluster_centers_[0][0], means.cluster_centers_[0][0], means.cluster_centers_[0][0], marker='p',c='c', )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    dot1 = ax.scatter3D([], [],[], color='r', s=1)
    dot2 = ax.scatter3D([], [],[], color='b', s=1) 
    fig.legend(handles=[dot1], labels=['abnormal sample'])
    fig.savefig('./3d_k-means.png')
