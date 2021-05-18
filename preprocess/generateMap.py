import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd 

class ClusterGenerator:
    def __init__(self, moduelNum, dataSource="../data_source/present_data.csv"):
        self.num = moduelNum
        self.source = dataSource
        self.classifications = None
        self.centers = None
        self.oriData = None
        self.clusterData = None

    def dataMap(self):
        None

    def loadData(self):
        dataf = pd.read_csv(self.source)
        f_dataf = dataf.dropna()
        self.oriData = f_dataf
        self.clusterData = self.oriData.copy()
        data_np = f_dataf.to_numpy(dtype="float32")
        self.data = data_np[:,1:3]
        print(self.data.shape)
        return None

    def showGrid(self):
        plt.figure(figsize=(15, 15))
        scatter1 = plt.scatter(x=self.data[:, 0], y=self.data[:, 1], s=50, c=self.classifications)
        scatter2 = plt.scatter(x=self.centers[:, 0], y=self.centers[:, 1], s=500, c='k', marker='^')
        plt.legend(*scatter1.legend_elements(),
                    loc="upper left", title="Ranking")
        plt.show()
    
    def kmeans(self, normalize=False, limit=200):

        """Basic k-means clustering algorithm.
        """
        # optionally normalize the data. k-means will perform poorly or strangely if the dimensions
        # don't have the same ranges.

        data = self.data
        k = self.num

        if normalize:
            stats = (data.mean(axis=0), data.std(axis=0))
            data = (data - stats[0]) / stats[1]
        
        # pick the first k points to be the centers. this also ensures that each group has at least
        # one point.
        centers = data[:k]

        for i in range(limit):
            # core of clustering algorithm...
            # first, use broadcasting to calculate the distance from each point to each center, then
            # classify based on the minimum distance.
            classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :])**2).sum(axis=1), axis=1)
            # next, calculate the new centers for each cluster.
            new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

            # if the centers aren't moving anymore it is time to stop.
            if (new_centers == centers).all():
                break
            else:
                centers = new_centers
        else:
            # this will not execute if the for loop exits on a break.
            raise RuntimeError(f"Clustering algorithm did not complete within {limit} iterations")
                
        # if data was normalized, the cluster group centers are no longer scaled the same way the original
        # data is scaled.
        # if normalize:
        #     centers = centers * stats[1] + stats[0]

        print(f"Clustering completed after {i} iterations")

        self.classifications = classifications
        self.centers = centers
        self.clusterData['cluster'] = classifications