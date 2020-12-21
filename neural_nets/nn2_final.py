import numpy as np
from sklearn import datasets
from sklearn import preprocessing

# neural net for regression
# 5 features, 1 hidden layer with 10 nodes
# leaky-relu activation, MSE loss function

# house_sk_data = datasets.load_boston()
# # size of training and cross validation sets
# m_train = 400
# m_cv = 506 - m_train

# either dataset works
house_sk_data = datasets.load_diabetes()
m_train = 400
m_cv = 442 - m_train

# training set
house_sk_train = preprocessing.scale(house_sk_data.data[:m_train, :5])
house_sk_train_bias = np.column_stack([np.ones((m_train, 1)), house_sk_train])

# CV set
house_sk_cv = preprocessing.scale(house_sk_data.data[m_train:, :5])
house_sk_cv_bias = np.column_stack([np.ones((m_cv, 1)), house_sk_cv])

# target set
house_sk_target = house_sk_data.target[:m_train]

# randomly initialize weights for first and second layer
weights = np.random.rand(10, 6)
weights2 = np.random.rand(1, 11)


# Forward and backward propagates for a given training example _i_
def propagate(i):
    global weights2, weights

    # learning rate
    # 0.0003 retains lowest MSE for CV set
    a = 0.0003

    # z for second layer and activation (leaky-ReLu)
    z2 = np.dot(weights, np.transpose(house_sk_train_bias[i, :]))
    a2 = np.where(z2 > 0, z2, 0.01 * z2)
    # add bias unit for second layer
    a2_bias = np.concatenate((np.ones(1), a2))
    # z for third layer and final activation (ReLu)
    z3 = np.dot(weights2, a2_bias)
    # regression, so hypothesis is linear
    hypot = z3

    # Backpropagates to accumulate partials and deltas
    # Delta for third layer (mean squared error)
    delta3 = hypot - house_sk_target[i]
    # derivative of error w.r.t weights, second layer
    layer2partials = np.multiply(delta3, a2_bias)

    # derivative of error w.r.t weights, first layer
    # stack this the # of input features there are + 1
    layer2partials_matrix = np.column_stack([layer2partials[1:] for n in range(6)])

    # stack this the number of hidden layer units there are
    feat_bias_matrix = np.row_stack([house_sk_train_bias[i, :] for n in range(10)])

    layer1partials = np.multiply(layer2partials_matrix, feat_bias_matrix)
    weights2 = weights2 - a * layer2partials
    weights = weights - a * layer1partials


# Train weights
for z in range(0, m_train):
    propagate(z)


# predict example in given set
def predict(j, pred_set):
    # z for second layer and activation (ReLu)
    z2 = np.dot(weights, np.transpose(pred_set[j, :]))
    # print('z2', z2)
    a2 = np.where(z2 > 0, z2, 0.01 * z2)
    print('a2', a2)
    # add bias unit for second layer
    a2_bias = np.concatenate((np.ones(1), a2))
    # z for third layer, no final activation for now
    z3 = np.dot(weights2, a2_bias)

    # calculate error
    sq_error = (np.square(z3 - house_sk_target[j]))
    print(f'Prediction: {z3}, actual: {house_sk_target[j]}, error: {sq_error}')
    return sq_error


tot_err_train = 0
# Predict in training example and see average error
for y in range(0, m_train):
    tot_err_train += predict(y, house_sk_train_bias)

print('=========CV-SET=========')
tot_err = 0
# Predict CV example and see average error
for x in range(0, m_cv):
    tot_err += predict(x, house_sk_cv_bias)

print('MSE training ($): ', tot_err_train / m_train)
print('MSE CV ($): ', tot_err / m_cv)
