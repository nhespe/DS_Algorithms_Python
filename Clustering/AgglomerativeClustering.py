""" """
# Realtive
from utils.distance import DISTANCE_FUNCTIONS

#External
import numpy as np

#Builtins
from typing import List
from copy import deepcopy
""" Agglomerative Clustering method """


class Cluster:
    def __init__(self, id, centroid, values=[]):
        self.centroid = centroid
        self.values = values
        self.id = id
        

class AgglomerativeClusterer:
    def __init__(self, k, distance_metric="euclidean"):
        self.n_clusters = k
        self.was_fit = False

        if distance_metric not in DISTANCE_FUNCTIONS:
            raise ValueError #ValueError(f"{distance_metric} not recognized")
        self.distance_metric = DISTANCE_FUNCTIONS.get(distance_metric)


    def fit(self, data):
        if not self.was_fit:
            self.was_fit = True

        clusters = [Cluster(idx, data_i, [data_i]) for idx, data_i in enumerate(data)]
        cluster_iterations = [deepcopy(clusters)]

        for i in range(len(data)-self.n_clusters):
            combined_clusters = self._combine_closest_clusters(clusters)
            cluster_iterations.append(combined_clusters)
            clusters = combined_clusters

        self.fit_clusters = cluster_iterations


    def get_cluster_iterations(self):
        """ Return a list of the outcomes of each cluster iteration- given in order"""
        if self.was_fit:
            return self.fit_clusters

        print("Algorithm Not Fit")
        return None


    def get_clusters(self):
        """ Return a list of the output Clusters """
        if self.was_fit:
            return self.fit_clusters[-1]

        print("Algorithm Not Fit")
        return None


    def _combine_closest_clusters(self, clusters):
        """ we need to compare i for j to find the two clusters that are closest """
        cluster_distances = np.full([len(clusters), len(clusters)], fill_value=np.inf)

        for idx, cluster in enumerate(clusters):
            for i in range(len(clusters) - idx - 2):
                target_index = i + idx + 1
                target_cluster = clusters[target_index]
                cluster_distances[idx, target_index] = self.distance_metric(cluster.centroid, target_cluster.centroid)

        min_value = cluster_distances.min()
        # take the first instance of the distances = min_value
        indices = np.argwhere(cluster_distances == min_value)[0]

        return self.merge_clusters(*tuple(indices), clusters=clusters)


    def merge_clusters(self, idx1, idx2, clusters):
        """ given two indices and a set of clusters, return a new clusters list, with the clusters at idx1 and idx2 merged

        *note* the new "merged" cluster.idx = idx1
        """
        c1, c2 = clusters[idx1], clusters[idx2]
        cluster_values = c1.values + c2.values
        new_centriod = np.array(c1.values + c2.values).mean(axis=0)
        
        merged_cluster = Cluster(centroid=new_centriod, values=cluster_values, id=c1.id)

        out =  [x for idx, x in enumerate(clusters) if idx not in [idx1, idx2]] + [merged_cluster]
        return out