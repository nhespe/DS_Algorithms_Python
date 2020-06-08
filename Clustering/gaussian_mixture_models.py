""" EM Algorithm for 1D, 2 Clusters"""
import math
import numpy as np
import random

class Gaussian:
	"""docstring for gaussian"""
	def __init__(self, mean, var):
		self.mean = mean
		self.var = var

	def get_mean(self, data): 
		self.mean = sum(data)/len(data)

	def get_var(self, data):
		self.var = sum([(x-self.mean)**2 for x in data])/len(data)

class Point:
	def __init__(self, location):
		self.location = location
		self.group = -1
	
class GMM_1d:
	def __init__(self, n_classes, maxiter=10):
		self.n_classes = 2
		self.n_dims = 1
		self.clusters = [None, None]
		self.maxiter = maxiter

	def fit(self, data, verbose=False):
		points = [Point(x) for x in data]
		prior_a, prior_b = 0.5, 0.5

		# instantiate Gaussians at the high and low of each 
		self.clusters[0] = Gaussian(np.min(data), 1)
		self.clusters[1] = Gaussian(np.max(data), 1)

		for i in range(self.maxiter):
			for point in points:
				# get cluster prob
				p_x_1 = self.p_x_given_gauss(point, self.clusters[0])
				p_x_2 = self.p_x_given_gauss(point, self.clusters[1])
				pb = self.p_gauss_given_point(p_x_1, p_x_2, prior_a, prior_b)
				pa = 1 - pb

				point.group = 2 if pb > pa else 1
			
			prior_a, prior_b = self.get_cluster_prior(points)

			# gaussian 1
			gaussian_points_1 = [x.location for x in points if x.group == 1]
			self.clusters[0].get_mean(gaussian_points_1)
			self.clusters[0].get_var(gaussian_points_1)

			# gaussian 2
			gaussian_points_2 = [x.location for x in points if x.group == 2]
			self.clusters[1].get_mean(gaussian_points_2)
			self.clusters[1].get_var(gaussian_points_2)

		if verbose:
			print([point.group for point in points])


	def p_x_given_gauss(self, point, gaussian):
		part1 = 1/math.sqrt(6.283 * (gaussian.var))
		numerator = -((point.location - gaussian.mean)**2)
		denominator = 2 * gaussian.var
		part2 = math.exp(numerator / float(denominator))
		return part1 * part2

	def p_gauss_given_point(self, x_given_1, x_given_2, p_1, p_2):
		numerator = x_given_2 * p_2
		denominator = numerator + (x_given_1 * p_1) 
		return numerator / float(denominator)

	def get_cluster_prior(self, data):
		p_a = sum([1 if x.group == 1 else 0 for x in data])/float(len(data))
		return p_a, 1-p_a

	def get_values(self, points):
		return [(point.group, point.location) for point in pointss]


# x = GMM_1d(2)
# x.fit([1,3,3.5,2,7,8,9,8.5,8.75,8.25,5,11], True)
