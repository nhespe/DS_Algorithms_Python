""" Distance Metrics """
import math

def euclidean(point_a, point_b):
	return math.sqrt(sum([(a-b)**2 for a, b in zip(point_a, point_b)]))

DISTANCE_FUNCTIONS = {
	"euclidean" : euclidean
}