import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, AffinityPropagation, MeanShift,estimate_bandwidth
import time
import seaborn as sns
import hdbscan as hd


sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha':0.25,'s':80,'linewidths':0,'cmap':'Spectral'}

#question 1
#loading cluster_data.npy set
data = np.load('cluster_data.npy')

#scatter plot of data set
plt.scatter(data.T[0],data.T[1], c ='b',**plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
#plt.show()
#establishing utility function
def plot_clusters(data, algorithm, args, kwds):
    starttime = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max()+1)
    colors = [palette[x] if x >=0 else (0.0,0.0,0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5,0.7, 'Clustering took {:.2f} s'.format(end_time-starttime),fontsize=14)
    plt.show()

#question 2
#finding the perfect k clusters form kmeans using the elbow method
sum_of_suared_distances = []
#print(data.shape)
kn = range(1,15)
for k in kn:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    sum_of_suared_distances.append(km.inertia_)

plt.plot(kn,sum_of_suared_distances,'bx-')
plt.xlabel('k')
plt.ylabel('sum of squared distances')
plt.title('elbow method for optimal k')
print("from the plot, our K cluster is 4")

#using n_clusters = 4, because its the optimal value to perform analysis
print("KMeans")
plot_clusters(data,KMeans,(),{'n_clusters':4})

bandwidth = estimate_bandwidth(data,quantile=0.2,n_samples=500)
print("Mean Shift")
plot_clusters(data,MeanShift,(),{'bandwidth':bandwidth})

print("AffinityPropagation")
plot_clusters(data,AffinityPropagation,(),{'damping':0.7,'max_iter':250,'random_state':0})

print("Spectral Clustering")
plot_clusters(data,SpectralClustering,(),{'n_clusters':4})

print("Agglomerative Clustering")
plot_clusters(data,AgglomerativeClustering,(),{'n_clusters':4})

print("HDBSCAN")
plot_clusters(data,hd.HDBSCAN,(),{'min_cluster_size':10})
