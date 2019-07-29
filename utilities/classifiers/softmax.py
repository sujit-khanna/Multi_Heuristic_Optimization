import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
	num_train = X.shape[0]
	num_classes = W.shape[1]
	loss = 0.0
	dW = np.zeros_like(W) # initialize gradient to 0
	for i in range(num_train):
		unorm_log_probs = np.dot(X[i], W)
		unorm_log_probs -= np.max(unorm_log_probs)
		probs = np.exp(unorm_log_probs) / np.sum(np.exp(unorm_log_probs))
		loss -= np.log(probs[y[i]])
		probs[y[i]] -= 1

		for j in range(num_classes):
			dW[:, j] += X[i, :] * probs[j]

	loss /= num_train
	dW /= num_train
	loss += 0.5 * reg * np.sum(W * W)
	dW += reg * W

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	num_train = X.shape[0]
	num_classes = W.shape[1]
	dW = np.zeros_like(W)
	unorm_log_probs = np.dot(X, W)
	unorm_log_probs -= np.max(unorm_log_probs, axis=1, keepdims=True)
	unorm_probs = np.exp(unorm_log_probs)
	probs = unorm_probs / np.sum(unorm_probs, axis=1, keepdims=True)

	corect_logprobs = -np.log(probs[np.arange(num_train), y])
	data_loss = np.sum(corect_logprobs) / num_train
	reg_loss = 0.5 * reg * np.sum(W * W)
	loss = data_loss + reg_loss
	probs[np.arange(num_train), y] -= 1
	probs /= num_train
	dW = np.dot(X.T, probs)

	return loss, dW

