import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) # numerical stability

    scores_exp = np.exp(scores)
    scores_sum = np.sum(scores_exp)
    # log_score_sum = np.log(scores_sum)
    # correct_class_score = scores[y[i]]
    # loss += -correct_class_score + log_score_sum
    # correct_class_score_exp = scores_exp[y[i]]
    scores_ratios = scores_exp / scores_sum
    # p_i = correct_class_score_exp / scores_sum
    loss -= np.log(scores_ratios[y[i]])

    dScores = scores_ratios
    dScores[y[i]] -= 1

    for j in xrange(num_classes): # (2,x)
      dW[:, j] += X[i, :] * dScores[j]

    # dW[i, :] += dScores * X[i]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  # Forward pass and loss
  scores = X.dot(W)
  scores -= X.max(axis=1)[:, np.newaxis]
  scores_exp = np.exp(scores)
  scores_sum = np.sum(scores_exp, axis=1)[:, np.newaxis]
  scores_ratios = scores_exp / scores_sum
  losses = -np.log(scores_ratios[np.arange(num_train), y])
  loss = np.sum(losses)

  # Compute gradient
  dScores = scores_ratios
  dScores[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, dScores)

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

