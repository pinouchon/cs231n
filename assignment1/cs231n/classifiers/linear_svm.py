import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margin_sum = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W) # (1,x)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes): # (2,x)
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        margin_sum += margin
        dW[:, y[i]] += -X[i]
        dW[:, j] += X[i]

  # (3)

  inv_num_train = 1 / num_train # (.)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  margin_mean = margin_sum * inv_num_train # (4)
  dW *= inv_num_train

  # Compute regularisation term
  reg_sum_w2 = reg * np.sum(W * W) # (5)

  # Add regularization to the loss.
  loss = margin_mean + reg_sum_w2 # (6)

  # Add loss derivative to dW
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W)
  scores_y = scores[np.arange(num_train), y]
  #print(scores.shape)
  #print(scores_y[:, np.newaxis].shape)
  margins = np.maximum(0, scores - scores_y[:, np.newaxis] + 1)
  margins[np.arange(num_train), y] = 0

  #loss = np.sum(margins, axis=1)
  #loss = np.sum(loss) / num_train
  loss = np.sum(margins) / num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin_mask = np.zeros(margins.shape)
  margin_mask[margins > 0] = 1
  sums = margin_mask.sum(axis=1)
  margin_mask[np.arange(num_train), y] -= sums.T
  dMargins = margin_mask / num_train
  dW += np.dot(X.T, dMargins)
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
