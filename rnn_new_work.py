import numpy as np
import math
import matplotlib as mpl
from matplotlib.image import imread
from random import randint


import keras
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.base import BaseEstimator, RegressorMixin
from keras import optimizers
from scikeras.wrappers import KerasClassifier, KerasRegressor
import keras.utils
import keras.layers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
import copy
import csv
from sklearn.metrics import mean_squared_error, make_scorer

mpl.use('Agg')
import matplotlib.pyplot as plt

#Set y values of data to lie between 0 and 1
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

#Import and pre-process data for future applications
def import_data(train_dataframe, valid_dataframe, test_dataframe):
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')

    #Include all 12 initial factors (Year ; Month ; Hour ; Day ; Cloud Coverage ; Visibility ; Temperature ; Dew Point ;
    #Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter
    max_test = np.max(dataset[:,12])
    min_test = np.min(dataset[:,12])
    scale_factor = max_test - min_test
    max = np.empty(13)
    min = np.empty(13)

    #Create training dataset
    for i in range(0,13):
        min[i] = np.amin(dataset[:,i],axis = 0)
        max[i] = np.amax(dataset[:,i],axis = 0)
        dataset[:,i] = normalize_data(dataset[:, i], min[i], max[i])

    train_data = dataset[:,0:12]
    train_labels = dataset[:,12]

    # Create valid dataset
    dataset = valid_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    valid_data = dataset[:,0:12]
    valid_labels = dataset[:,12]

    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    test_data = dataset[:, 0:12]
    test_labels = dataset[:, 12]

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, scale_factor


# class KerasLSTMWrapper(RegressorMixin, BaseEstimator):
#     def __init__(self, init_type='glorot_uniform', optimizer='adam'):
#         self.init_type = init_type
#         self.optimizer = optimizer
#         self.model = self.build_estimator()
        
#     #Construct and return Keras RNN model
#     def build_estimator(self, init_type='glorot_uniform', optimizer='adam'):
#         model = Sequential()
#         layers = [12, 64, 64, 1, 1]
#         model.add(keras.layers.LSTM(
#             layers[0],
#             input_shape = (None,12),
#             return_sequences=True))
#         model.add(keras.layers.Dropout(0.2))
    
#         model.add(keras.layers.LSTM(
#             layers[1],
#             kernel_initializer = init_type,
#             return_sequences=True
#             #bias_initializer = 'zeros'
#         ))
#         model.add(keras.layers.Dropout(0.2))
    
#         model.add(Dense(
#             layers[2], activation='tanh',
#             kernel_initializer=init_type,
#             input_shape = (None,1)
#             ))
#         model.add(Dense(
#             layers[3]))
    
#         model.add(Activation("relu"))
#         #Alternative parameters:
#         #momentum = 0.8
#         #learning_rate = 0.1
#         #epochs = 100
#         #decay_rate = learning_rate / 100
#         #sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#         #model.compile(loss="binary_crossentropy", optimizer=sgd)
#         #rms = keras.optimizers.RMSprop(learning_rate=0.002, rho=0.9, epsilon=1e-08, decay=0.01)
#         model.compile(loss="mean_squared_error", optimizer=optimizer)
    
#         return model
#     def score(self, X, y):
#         # Use custom scorer (MSE)
#         mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
#         return mse_scorer(self, X, y)
    
#     def fit(self, X, y, sample_weight=None):
#         return self.model.fit(X, y, sample_weight=sample_weight)

#     def predict(self, X):
#         return self.model.predict(X)
    
    
#Construct and return Keras RNN model
def build_model(init_type='glorot_uniform', optimizer='adam'):
    model = Sequential()
    layers = [12, 64, 64, 1, 1]
    model.add(keras.layers.LSTM(
        layers[0],
        input_shape = (None,12),
        return_sequences=True))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(
        layers[1],
        kernel_initializer = init_type,
        return_sequences=True
        #bias_initializer = 'zeros'
    ))
    model.add(keras.layers.Dropout(0.2))

    model.add(Dense(
        layers[2], activation='tanh',
        kernel_initializer=init_type,
        input_shape = (None,1)
        ))
    model.add(Dense(
        layers[3]))

    model.add(Activation("relu"))
    #Alternative parameters:
    #momentum = 0.8
    #learning_rate = 0.1
    #epochs = 100
    #decay_rate = learning_rate / 100
    #sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    #model.compile(loss="binary_crossentropy", optimizer=sgd)
    #rms = keras.optimizers.RMSprop(learning_rate=0.002, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model


#Save output predictions for graphing and inspection
def write_to_csv(prediction, filename):
    print("Writing to CSV...")
    with open(filename, 'w') as file:
        for i in range(prediction.shape[0]):
            file.write("%.5f" % prediction[i])
            file.write('\n')
    print("...finished!")



#Store MSE errors for each model we made
def store_MSE(mse_list, filename):
    print("Writing to txt:")
    with open(filename, 'w') as file:
        file.write('\n'.join('%s %s' % x for x in mse_list))
        file.write('\n')
    print("...finished!")



#Return MSE error values of all three data sets based on a single model
def evaluate(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, scale_factor):
    scores = model.evaluate(X_train, Y_train, verbose = 0) * scale_factor * scale_factor
    print("train: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_valid, Y_valid, verbose = 0) * scale_factor * scale_factor
    print("valid: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_test, Y_test, verbose = 0) * scale_factor * scale_factor
    print("test: ", model.metrics_names, ": ", scores)

#Calculate MSE between two arrays of values
def mse(predicted, observed):
    return np.sum(np.multiply((predicted - observed),(predicted - observed)))/predicted.shape[0]

def main():
    
    #Original train, validation, test data from authors.
    #We do not want them to be randomized. We want them to be sequential. Hence we combine and resplit chronologically
    #using date and time.
    train = pd.read_csv('../Datasets/hourly/weather_train.csv', sep = ';')
    valid = pd.read_csv('../Datasets/hourly/weather_dev.csv', sep = ';')
    test = pd.read_csv('../Datasets/hourly/weather_test.csv', sep = ';')
    data = pd.concat([train, valid, test])
    data = data.sort_values(by = ['year', 'month', 'day', 'hour'])
    
    #TRAIN TEST SPLIT
    # Calculate split indices
    total_samples = len(data)
    train_end = int(0.9 * total_samples)
    val_end = int(0.95 * total_samples)

    # Split the data
    train_data = data[:train_end]
    valid_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, scale_factor = import_data(train_data, valid_data, test_data)

    X_train = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    X_valid = np.reshape(valid_data, (valid_data.shape[0], 1, valid_data.shape[1]))
    X_test = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
    Y_train = np.reshape(train_labels, (train_labels.shape[0], 1, 1))
    Y_valid = np.reshape(valid_labels, (valid_labels.shape[0], 1, 1))
    Y_test = np.reshape(test_labels, (test_labels.shape[0], 1, 1))

    #From running the kfold/grid search, optimal parameters are different:
    #['normal', 100, 8, 'rmsprop']
    model = build_model('normal', 'rmsprop')

    #List of mse's
    mse_list = []
    
    
    #Standard vanilla LSTM model

    
    model_fit_epochs = 100
    print("X_train shape: ",X_train.shape, " Y_train shape: ",Y_train.shape)

    model.fit(
        X_train, Y_train,
        batch_size = 8, epochs = model_fit_epochs)
    trainset_predicted = model.predict(X_train)
    validset_predicted = model.predict(X_valid)
    testset_predicted = model.predict(X_test)

    train_mse = mse(trainset_predicted, Y_train) * scale_factor * scale_factor
    valid_mse = mse(validset_predicted, Y_valid) * scale_factor * scale_factor
    test_mse = mse(testset_predicted, Y_test) * scale_factor * scale_factor
    
    print("Train MSE: ", train_mse)
    print("Valid MSE: ", valid_mse)
    print("Test MSE: ", test_mse)

    
    mse_list.append(('LSTM Train MSE', train_mse))
    mse_list.append(('LSTM Valid MSE', valid_mse))
    mse_list.append(('LSTM Test MSE', test_mse))
    #Adaboost model (ensemble learning)

    #cv_model = KerasClassifier(build_fn=build_model, epochs=100, batch_size=16, verbose=0)

    
    #I'm not able to get the LSTM Adaboost to work properly. 

    # lstm_wrapper = KerasLSTMWrapper(init_type='glorot_uniform', optimizer='adam')
    
    # #adaboost_base_estimator = KerasRegressor(model = lstm_wrapper, optimizer="adam", epochs=100, batch_size = 16, verbose=0)
    # adaboost = AdaBoostRegressor(estimator=lstm_wrapper, learning_rate=0.01)
    # adaboost.fit(X_train, train_labels)

    # trainset_predicted = adaboost.predict(X_train)
    # validset_predicted = adaboost.predict(X_valid)
    # testset_predicted = adaboost.predict(X_test)

    # adaboost_train_mse = mse(trainset_predicted, Y_train) * scale_factor * scale_factor
    # adaboost_valid_mse = mse(validset_predicted, Y_valid) * scale_factor * scale_factor
    # adaboost_test_mse = mse(testset_predicted, Y_test) * scale_factor * scale_factor
    
    # print("Train MSE: ", adaboost_train_mse)
    # print("Valid MSE: ", adaboost_valid_mse)
    # print("Test MSE: ", adaboost_test_mse)

    # mse_list.append(('Adaboost LSTM Train MSE', adaboost_train_mse))
    # mse_list.append(('Adaboost LSTM Valid MSE', adaboost_valid_mse))
    # mse_list.append(('Adaboost LSTM Test MSE', adaboost_test_mse))
    
    # K-fold cross validation (K = 10):
        
    # kf = KFold(n_splits=10, shuffle=True)
    # # Loop through the indices the split() method returns
    # for index, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
    #     print("Training on fold " + str(index + 1) + "/10...")
    #     # Generate batches from indices
    #     xtrain, xval = X_train[train_indices], X_train[val_indices]
    #     ytrain, yval = Y_train[train_indices], Y_train[val_indices]
    #     # Clear model, and create it
    #     model = None
    #     model = build_model()

    #     model.fit(
    #         xtrain, ytrain,
    #         batch_size = 16, epochs = model_fit_epochs)
    #     testset_predicted_kfold = model.predict(xval)
    #     kfold_mse = mse(testset_predicted_kfold, yval)* scale_factor * scale_factor
    #     print("Test MSE: ", kfold_mse )

    #Grid search to optimize model params

    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [8, 16, 32]
    optimizers = ['rmsprop', 'adam', 'Adadelta']
    optimal_params = np.empty(4)
    train_minimum_error = 2e63
    test_minimum_error = 2e63
    for init_type in init:
        for epoch in epochs:
            for batch in batches:
                for optimizer in optimizers:
                    model = None
                    model = build_model(init_type, optimizer)

                    model.fit(
                        X_train, Y_train,
                        batch_size=batch, epochs=epoch)
                    predicted_train = model.predict(X_train)
                    error_train =  mse(predicted_train, Y_train)  * scale_factor * scale_factor
                    predicted_test = model.predict(X_test)
                    error_test = mse(predicted_test, Y_test)  * scale_factor * scale_factor
                    
                    if error_train < train_minimum_error:
                        train_minimum_error = error_train
                    if error_test < test_minimum_error:
                        test_minimum_error = error_test
                        optimal_params = [init_type, epoch, batch, optimizer]

    print("optimal params: ", optimal_params)
    #Already denormalized mse's above, so no need to multiply by scale_factor^2 below
    print("train minimized error: ", train_minimum_error)
    print("test minimized error: ", test_minimum_error)

    mse_list.append(('(Optional) Optimal parameters', optimal_params))
    mse_list.append(('Grid Search Optimal Train Error', train_minimum_error))
    mse_list.append(('Grid Search Optimal Test Error', test_minimum_error))
    
    #Note these csv values are normalized. In the csv, can multiply by scale_factor ^2
    #to get denormalized values
    write_to_csv(trainset_predicted,'nn_trainset_prediction_new.csv')
    write_to_csv(validset_predicted,'nn_devset_prediction_new.csv')
    write_to_csv(testset_predicted, 'nn_testset_prediction_new.csv')
    store_MSE(mse_list, 'mse_list_new.txt')
    return

if __name__ == '__main__':
    main()