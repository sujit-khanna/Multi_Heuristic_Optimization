import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):

	delta = 1.0
	loss = 0.0
	dW = np.zeros(W.shape)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	for i in range(num_train):
		scores = np.dot(X[i], W)
		correct_class_score = scores[y[i]]
		for j in range(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + delta
			if margin > 0:
				loss += margin
				dW[:, y[i]] -= X[i, :]
				dW[:, j] += X[i, :]


	loss /= num_train
	dW /= num_train
	loss += 0.5 * reg * np.sum(W*W)
	dW += reg*W

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	delta = 1.0
	dW = np.zeros(W.shape)
	num_train = X.shape[0]
	scores = np.dot(X, W)
	correct_class_score = scores[np.arange(num_train), y]
	margins = np.maximum(0, scores.T - correct_class_score + delta)
	loss = (np.sum(margins) - num_train) / num_train
	loss += 0.5 * reg * np.sum(W*W)
	slopes = np.zeros((margins.shape))
	slopes[margins>0] = 1
	slopes[y, range(num_train)] -= np.sum(margins>0, axis=0)
	dW = np.dot(X.T, slopes.T) / float(num_train)
	dW += reg * W
	return loss, dW


def svm_loss_vectorized_population(W, X, y, reg):
    delta = 1.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    scores = np.dot(X, W)
    correct_class_score = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores.T - correct_class_score + delta)
    loss = (np.sum(margins) - num_train) / num_train
    loss += 0.5 * reg * np.sum(W*W)
    objective = (1/loss)
    return loss, objective
