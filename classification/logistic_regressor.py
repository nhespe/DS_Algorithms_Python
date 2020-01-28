""" Coding Linear Regression """
import numpy as np
import random
import math

from matplotlib import pyplot as plt

INTERS = 100000
LEARNING_RATE = 0.001

x = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [[1], [1], [1], [0], [0]]

class LogisticReg_Trained:
	def __init__(self):
		self.n = None
		self.betas = None
		self.trained = False

	def fit_model(self, data, labels):
		loss = []
		self.betas = np.asarray([random.randint(0,5)/5.0 for i in range(data.shape[1])]).reshape(-1, 1)
		self.trained = True
		for i in range(INTERS):
			multiplied_values = np.dot(data, self.betas)
			y = [1/(1 + (math.exp(-1 * x)))	for x in multiplied_values]	
			avg_cost = np.asarray([LogisticReg_Trained.get_cost(y_i,lab) for y_i,lab in zip(y, labels)])/len(labels)

			loss.append(abs(sum(avg_cost)))
			gradient = np.dot(-data.T, avg_cost)
			gradient_lr = gradient * LEARNING_RATE
			self.betas = self.betas - gradient_lr

		print("Betas", list(self.betas))
		return loss

	@classmethod
	def get_cost(self, y_pred, label):
		return (-label * math.log(y_pred)) + ((1-label)*math.log(1 - y_pred))

	def predict_proba(self, data):
		if self.trained:
			multiplied_values= np.dot(data, self.betas)
			return [1/(1 + (math.exp(-1 * x))) for x in multiplied_values]	
		else:
			print("Classifier Not Fitted")

	def predict(self, data):
		if self.trained:
			multiplied_values= np.dot(data, self.betas)
			preds_proba =  [1/(1 + (math.exp(-1 * x))) for x in multiplied_values]	
			return [1 if x >= .5 else 0 for x in preds_proba]
		else:
			print("Classifier Not Fitted")


lr = LogisticReg_Trained()
losses = lr.fit_model(np.asarray(x), np.asarray(y))
indices = [i for i in range(len(losses))]

print(lr.predict(np.asarray(x)))

plt.scatter(indices, losses)
plt.title("Abs(avg_cost)(Loss) x Iteration")
plt.show()

