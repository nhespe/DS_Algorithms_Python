""" Coding Linear Regression """
import numpy as np
import random

ITERS = 100
LEARNING_RATE = 0.0001

x = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [[1], [2], [3], [4], [5]]
 
class LinearReg_Trained:
	def __init__(self):
		self.n = None
		self.betas = None
		self.trained = False

	def fit_model(self, data, labels):
		self.betas = np.asarray([random.randint(0,5)/5.0 for i in range(data.shape[1])]).reshape(-1, 1)
		self.trained = True
		for i in range(ITERS):
			y = np.dot(data, self.betas)
			errors = (labels - y)/len(y) # get avg error

			gradient = np.dot(-data.T, errors) # Basically x(xb - y) = 0
			gradient_lr = gradient * LEARNING_RATE
			self.betas = self.betas - gradient_lr

		print(list(y), self.betas)

	def classify(self, data):
		if self.trained:
			return np.dot(data, self.betas)
		else:
			print("Classifier Not Fitted")


class LinearReg_Calculated:
	def __init__(self):
		self.n = None
		self.betas = None
		self.trained = False

	def fit_model(self, data, labels):
		""" Solve For xTx**-1 * xTy """
		self.trained = True

		XtX = np.dot(np.transpose(data),data)
		Xty = np.dot(np.transpose(data),labels)
		self.betas = np.linalg.solve(XtX,Xty) 

	def classify(self, data):
		if self.trained:
			return np.dot(data, self.betas)
		else:
			print("Classifier Not Fitted")


lr = LinearReg_Trained()
lr.fit_model(np.asarray(x), np.asarray(y))
print(lr.classify(np.asarray(x)))
