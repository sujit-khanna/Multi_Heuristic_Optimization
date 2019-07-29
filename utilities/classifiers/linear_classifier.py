import numpy as np
from utilities.classifiers.linear_svm import *
from utilities.classifiers.softmax import *

class LinearClassifier(object):

	def __init__(self):
		self.W = None

	def train(self, X, y, learning_rate=0.0001, reg=0.001, num_iters=1000, batch_size=200, verbose=False):

		num_train, dim = X.shape
		num_classes = np.max(y) + 1

		if self.W is None:
			self.W = 0.001 * np.random.randn(dim, num_classes)
		loss_history = []
		for it in range(num_iters):
			X_batch = None
			y_batch = None
			mask = np.random.choice(num_train, batch_size, replace=True)
			X_batch = X[mask, :]
			y_batch = y[mask]
			loss, grad = self.loss(X_batch, y_batch, reg)
			#print('loss and gradients are: ')
			#print(loss, grad)
			loss_history.append(loss)
			self.W -= learning_rate * grad

			if verbose and it % 100 == 0:
				print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

		return loss_history

	def ensemble_train(self):
		'''Inherit from EO optimizer; export this library to the ensemble repo'''
		pass

	def predict(self, X):

		y_pred = np.zeros(X.shape[1])
		y_pred = np.argmax(X.dot(self.W), axis=1)

		return y_pred

	def loss(self, X_batch, y_batch, reg):
		pass

class LinearSVM(LinearClassifier):
	def loss(self, X_batch, y_batch, reg):
		return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
		#return svm_loss_naive(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
	def loss(self, X_batch, y_batch, reg):
		return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)		