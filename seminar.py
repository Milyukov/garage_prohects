import pandas as pd
import numpy as np
import pickle
import time

from datetime import datetime

np.random.seed(0)

def evaluate_loss(Y, Y_pred):
    return np.sum((Y - Y_pred) ** 2)

def scale_data(X):
    X_scaled = X - np.mean(X, axis=0)
    X_scaled /= np.std(X_scaled, axis=0)
    return X_scaled

def add_bias(X):
    bias = -1 * np.ones((X.shape[0], 1))
    return np.concatenate((bias, X), axis=1)

def sgd(X, Y, epochs, step):
    W = np.random.random((X.shape[1], Y.shape[1])) - 0.5
    # calculate gradient
    # W = np.inv(X.T.dot(X)).dot(X).dot(Y)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # get prediction
    Y_pred = X.dot(W)
    # evaluate loss
    loss = evaluate_loss(Y, Y_pred)
    print('Loss: {}'.format(loss))
    return W

def split_train_test(X, Y, ratio=0.1):
    indexes = np.random.randint(0, X.shape[0] - 1, X.shape[0])
    number_of_train = int((1 - ratio) * X.shape[0])
    X_train = X[indexes[:number_of_train], :]
    Y_train = Y[indexes[:number_of_train], :]
    X_test = X[indexes[number_of_train:], :]
    Y_test = Y[indexes[number_of_train:], :]
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    # read data
    df = pd.read_csv('./moscow_dataset_2020.csv')
    # parse columns
    features_interpretation = df.columns[:-1]
    raw_features = df[features_interpretation]
    values_interpretation = df.columns[-1]
    # convert string values to integers
    string_values = np.unique(df[features_interpretation[0]].values)
    int_values = {index: value for index, value in enumerate(string_values)}
    for key, val in int_values.items():
        df = df.replace(val, key)
    # generate X-matrix of samples/features
    X = df[features_interpretation].values
    X = scale_data(X)
    X = add_bias(X)

    # generate Y-vector of values to predict
    Y = df[values_interpretation].values
    Y = Y.astype(np.float) / 1e6
    Y = np.expand_dims(Y, axis=-1)

    # split data for testing and training
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y)

    # SGD:
    W = sgd(X_train, Y_train, epochs=1000, step=1e-3)

    # validation
    Y_pred = X_test.dot(W)
    acc = np.sqrt(np.sum((Y_pred - Y_test) ** 2) / float(Y_test.shape[0]))
    print('STD: {:.2f} * 1e6'.format(acc))

    now = datetime.now()  # current date and time
    date_str = now.strftime("%m_%d_%Y_%H_%M_%S")
    with open('./weights_{}.pickle'.format(date_str), 'wb') as f:
        pickle.dump(W, f)
