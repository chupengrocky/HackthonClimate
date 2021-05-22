import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import normalize
class ClusterGenerator:
    def __init__(self, moduelNum, dataSource="../data_source/present_data.csv"):
        self.num = moduelNum
        self.source = dataSource
        self.classifications = None
        self.centers = None
        self.oriData = None

        self.meanShiftData = None
        self.sampleData = None

    def dataMap(self):
        None

    def loadData(self):
        dataf = pd.read_csv(self.source)
        f_dataf = dataf.dropna()
        self.oriData = f_dataf

        # Make a copy of outPut data 
        self.meanShiftData = self.oriData.copy()
        self.sampleData = self.oriData.copy()

        self.data_np = f_dataf.to_numpy(dtype="float32")        
        self.location_data = self.data_np[:,1:3]
        print(self.location_data.shape)
        return None

    def saveClusterFig(self):
        plt.figure(figsize=(8, 8))
        scatter1 = plt.scatter(x=self.location_data[:, 0], y=self.location_data[:, 1], s=50, c=self.classifications)
        print(self.location_data.shape)
        plt.scatter(x=self.centers[:, 0], y=self.centers[:, 1], s=500, c='k', marker='^')
        plt.legend(*scatter1.legend_elements(),
                    loc="upper left", title="Ranking")

        plt.xlim([30, 45])
        plt.ylim([235.5,246])
        plt.savefig("fig/{}_cluster.png".format(self.num))
        
    def saveSingleFig(self,indx):
        plt.figure(figsize=(8, 8))
        la,lo = self.centers[indx]
        points = self.sampleData[self.sampleData["model_{}".format(indx)]==1].to_numpy(dtype="float32")
        plt.scatter(x=points[:, 1], y=points[:, 2], s=50,c=['red'])
        scatter2 =plt.scatter(x=self.centers[:, 0], y=self.centers[:, 1], s=500, c=range(len(self.centers)), marker='^',)
        plt.title("Model_{} decay:{}".format(indx,self.decay))
        plt.legend(*scatter2.legend_elements(),
                    loc="upper left", title="Ranking")

        plt.xlim([30, 45])
        plt.ylim([235.5,246])
        plt.savefig("fig/mode_{}_data_distribution.png".format(indx))
    
    def kmeans(self, normalize=False, limit=200):

        """Basic k-means clustering algorithm.
        """
        # optionally normalize the data. k-means will perform poorly or strangely if the dimensions
        # don't have the same ranges.

        data = self.location_data
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
        if normalize:
            centers = centers * stats[1] + stats[0]

        print(f"Clustering completed after {i} iterations")
        
        
        self.classifications = classifications
        self.centers = centers
        for i in range(len(centers)):
            col = np.zeros(len(classifications))
            col[classifications==i]=1
            self.meanShiftData['model_{}'.format(i)]=col
        self.meanShiftData.to_csv('temp/meanShiftData.csv')
        self.saveClusterFig()



    def randomSample(self, decay=2, size=200):
        self.decay = decay
        for indx in range(len(self.centers)):
            la, lo = self.centers[indx]
            distence = np.sqrt((self.data_np[:,1]-la)**2+(self.data_np[:,2]-lo)**2)
            probability = 1./(distence**decay)
            probability = probability/sum(probability)
            # print(sum(probability))
            # probability1 = normalize((1./(distence*decay)).reshape(1,-1))
            # probability2 = normalize((1./(distence)).reshape(1,-1))
            # print(probability1,probability2)
            # probability = np.exp(-distence*decay)
            # probability = scipy.stats.norm.cdf(1./(distence*decay)) 
            # print(min(probability),max(probability))
            # model_selection = np.random.binomial(1, p=probability.reshape(-1))
            # print(range(self.data_np.shape[0]))
            print(np.arange(self.data_np.shape[0]).shape,probability.shape)
            model_selection = np.random.choice(np.arange(self.data_np.shape[0]), size, replace=False, p=probability)

            model_mask = np.zeros(probability.shape[0])
            model_mask[model_selection] = 1
            print(sum(model_mask))
            self.sampleData['model_{}'.format(indx)] = model_mask
            print("model_{} select {} sample".format(indx,size))
            
            self.saveSingleFig(indx)
        self.sampleData.to_csv('temp/sampleData.csv')

