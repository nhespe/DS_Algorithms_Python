"""Code for KMeans Clustering Algorithm -> WIP"""
from sklearn import datasets
import random

from typing import List, Callable
import numpy as np
import math
import collections


class KMeansClassifier:
	def __init__(self, k, m_iter=50):
		self.k = k
		self.m_iter = m_iter

	def fit(self, data, init_method: str = "random") -> int:
		""" given a chunk of data first we classify/fit all points """
		initialization_function = self._validate_init_method(init_method.lower())
		cluster_locs = initialization_function(self.k, data)
		print("starting locations", cluster_locs)

		for itr in range(self.m_iter):
			assigned_clusters = [self._get_cluster(point, cluster_locs) for point in data]
			print(assigned_clusters)

			reassigned_cluster_centriods = self._get_cluster_centriods(assigned_clusters, data)
			print(reassigned_cluster_centriods)

			if all([np.array_equal(x,y) for x,y in zip(reassigned_cluster_centriods, cluster_locs)]):
				print(f"Exit Criterion Met, Exiting at Iter: {itr}")
				self.fit_data, self.centriods = assigned_clusters, cluster_locs
				return 1

			cluster_locs = reassigned_cluster_centriods

		self.fit_data, self.centriods = assigned_clusters, cluster_locs
		return 1


	def _get_cluster_centriods(self, assigned_clusters, data):
		new_clust_locs = []
		
		for i in range(self.k):
			subset_cluster = np.array([data_i for point, data_i in zip(assigned_clusters, data) if point==i])

			if subset_cluster.size:
				new_clust_locs.append(np.mean(subset_cluster, axis=0))
			else:
				new_clust_locs.append([0.0]*data.shape[1])

		return new_clust_locs

	def _get_cluster(self, point, cluster_locations):
		clulster_distances = []
		for cluster_loc in cluster_locations:
			clulster_distances.append(KMeansClassifier.euclidean_distance(point, cluster_loc))

		return np.argmin(clulster_distances)

	def _validate_init_method(self, method: str):
		if method in VALID_METHODS:
			return VALID_METHODS[method]
		else:
			raise ValueError(f"Invalid method provided: {method}")

	@classmethod
	def euclidean_distance(self, point_a, point_b):
		return math.sqrt(sum([(a-b)**2 for a, b in zip(point_a, point_b)]))

	@classmethod
	def _random_points(cls, k, data) -> List[int]:
		#max_, min_ = np.max(data[:,0]) + 1, np.min(data[:,0])
		min_, max_ = 0, 10
		depth = data.shape[1]
		return [np.array([random.randint(int(min_), int(max_)) for i_d in range(depth)]) for ki in range(k)]


VALID_METHODS = {"random": KMeansClassifier._random_points}


# Import some data to play with
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target

classifier = KMeansClassifier(len(collections.Counter(y)))
classifier.fit(x, "random")

print(y)
print([[x,y] for x,y in zip(classifier.fit_data, y)])


