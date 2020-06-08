""" Work in Progress ** Code for Naive Bayes Classifier - Multiclass + Variable Feature Numbers """

from collections import Counter

import numpy as np

class NaiveBayes: 
	def __init__(self, smooth=False):
		self.smooth = smooth

	def fit(self, data, labels):
		n_classes = len(set(labels))

		# develop class priors
		class_priors = Counter(labels)

		feat_priors = dict()

		for i in n_classes:
			class_dict = dict()
			for feat_index in data.shape[1]:
				class_subset = [data_i, class_ for data_i, class_ in zip(data, labels) if class_ == i]
				class_dict[feat_index] = Counter(class_subset[:feat_index])

		self.n_feats = data.shape[1]
		self.n_classes = n_classes
		self.feature_priors = feat_priors
		self.class_priors = class_priors


	def predict(self, data):
		prediction = []
		for row in data:
			prediction.append(self.predict_row(data))

		return prediction

	def predict_row(self, row):
		ret_array = []
		class_probabilities = []
		for class_i in range(self.n_classes):
			priors_feat_given_class = []
			for feat in n_feats:
				smoother_value = 1 if self.smooth else 0
				p_feat_class = feature_priors[class_i][feat].get(row, smoother_value) / sum(feature_priors[class_i][feat].values())
				priors_feat_given_class.append(p_feat_class)

			priors_feat_given_class.append(self.class_priors[class_i])
			class_probabilities.append(np.prod(priors_feat_given_class))

		for class_i in range(self.n_classes):
			ret_array.append(class_probabilities[i] / np.prod([cl for idx, cl in enumerate(class_probabilities) if idx != class_i]))

		return ret_array
