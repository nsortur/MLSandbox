import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from scipy.special import expit

# Neural net for regression on Boston house prices
# using features:
# B 1000(Bk - 0.63)^2
# LSTAT % lower status of the population
# scaled for zero mean and unit variance

# 2 input features
# 1 hidden layer, 2 nodes, sigmoid (expit) activation
# MSE loss function, no regularization

house_sk_data = datasets.load_boston()
# size of training and cross validation sets
m_train = 400
m_cv = 506 - m_train

# training set
house_sk_train = preprocessing.scale(house_sk_data.data[:m_train, 10:12])
house_sk_train_bias = np.column_stack([np.ones((m_train, 1)), house_sk_train])

# CV set
house_sk_cv = preprocessing.scale(house_sk_data.data[m_train:, 10:12])
house_sk_cv_bias = np.column_stack([np.ones((m_cv, 1)), house_sk_cv])

# target set
house_sk_target = house_sk_data.target[:m_train]

# randomly initialize weights for first and second layer
weights = np.random.rand(2, 3)
weights2 = np.random.rand(1, 3)


# Forward and backward propagates for a given training example __i__
def propagate(i):
    global weights2, weights

    # learning rate (0.04 best for CV set)
    a = 0.04

    # z for second layer and activation (ReLu)
    z2 = np.dot(weights, np.transpose(house_sk_train_bias[i, :]))
    a2 = expit(z2)
    # add bias unit for second layer
    a2_bias = np.concatenate((np.ones(1), a2))
    # z for third layer
    z3 = np.dot(weights2, a2_bias)
    # no final activation for now
    hypot = z3

    # Backpropagates to accumulate partials and deltas
    # Delta for third layer (mean squared error)
    delta3 = hypot - house_sk_target[i]
    # derivative of error w.r.t weights, second layer
    layer2partials = np.multiply(delta3, a2_bias)

    # derivative of error w.r.t weights, first layer
    delta2 = np.multiply(delta3, a2)
    delta2_matrix = np.column_stack([delta2, delta2, delta2])
    feat_bias_matrix = np.row_stack([house_sk_train_bias[i, :], house_sk_train_bias[i, :]])
    layer1partials = np.multiply(delta2_matrix, feat_bias_matrix)

    weights2 = weights2 - a * layer2partials
    weights = weights - a * layer1partials


# Train weights
for i in range(0, m_train):
    propagate(i)


# predict example __j__ in given set __pred_set__
def predict(j, pred_set):
    # z for second layer and activation (sigmoid)
    z2 = np.dot(weights, np.transpose(pred_set[j, :]))
    a2 = expit(z2)

    # add bias unit for second layer
    a2_bias = np.concatenate((np.ones(1), a2))
    # z for third layer, no final activation for now
    z3 = np.dot(weights2, a2_bias)

    # calculate error
    sq_error = np.square(z3 - house_sk_target[j])
    # print(f'Prediction: {z3}, actual: {house_sk_target[j]}, error: {sq_error}')
    return sq_error


tot_err = 0
# Predict CV example and see average error
for j in range(0, m_cv):
    tot_err += predict(j, house_sk_cv_bias)

print('Average squared CV error ($): ', tot_err / m_cv)

tot_err = 0
# Predict in training example and see average error
for j in range(0, m_train):
    tot_err += predict(j, house_sk_train_bias)

print('Average squared training error ($): ', tot_err / m_train)
