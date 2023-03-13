import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



embedding = np.load('./tsne_3d.npy')
print(embedding.shape)
_max, _min = np.max(embedding), np.min(embedding)
embedding = (embedding - _min) / (_max - _min)
mean_vector = [np.mean(embedding[0]), np.mean(embedding[1]), np.mean(embedding[2])]

distance = []
for item in embedding:
    d = (item[0]-mean_vector[0])**2+(item[1]-mean_vector[1])**2+(item[2]-mean_vector[2])**2
    d = np.sqrt(d)
    distance.append(d)
    
distance = np.array(distance)

flierprops = dict(marker='_', markeredgecolor='red')
plt.figure(figsize=(3, 4))
fig = sns.boxplot(data=distance, width=0.01, linewidth=1, flierprops=flierprops)
fig.set_xlim(-0.1,0.1)
fig.set_ylim(0,1)
# sns.swarmplot(data=distance, color="grey")
fig = fig.get_figure()
fig.savefig('./box_plot.png')
    

# plt.figure()
# plt.scatter(distance, zeros, marker='o')

# plt.savefig('./box_plot.png')