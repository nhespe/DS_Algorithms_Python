""" K-Nearest Neighbors Classifier"""
import math
import numpy as np

from collections import Counter

from sklearn import datasets

class KNN_Classifier:
	def __init__(self, k):
		self.k_nearest = k

	def fit(self, data, y):
		if isinstance(data, np.ndarray):
			self.data = data
		elif isinstance(data, list):
			self.data = np.array(data)
		else:
			self.data = None

		self.labels = y

	def get_n_nearest_indices(self, point):
		n = self.k_nearest
		distances = np.array([KNN_Classifier._euclidean_distance(point, row) for idx, row in enumerate(self.data)])
		return list(np.argpartition(distances, n)[:n])

	def classify(self, point):
		indices = knn.get_n_nearest_indices(x[50])
		element_labels = knn.get_n_elements_by_index(indices)
		return Counter(element_labels).most_common(1)[0][0]

	def get_n_elements_by_index(self, indices):
		return self.labels[indices]

	@classmethod
	def _euclidean_distance(self, point_a, point_b):
		return math.sqrt(sum([(a-b)**2 for a, b in zip(point_a, point_b)]))


# Import some data to play with
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target

knn = KNN_Classifier(5)
knn.fit(x,y)

# Target Label for test
print(y[50])

# By Step
indices = knn.get_n_nearest_indices(x[50])
element_labels = knn.get_n_elements_by_index(indices)
print("Outcome Based on Most Similar Elements", Counter(element_labels).most_common(1)[0][0])

# Classify 
print("Outcome Based on Most Similar Elements", knn.classify(x[50]))
