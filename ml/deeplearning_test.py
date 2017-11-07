from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectPercentile


from sklearn import svm
import mglearn
import sklearn

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 10
img_size = 28*28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32')/255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32')/255

print(X_train.shape, X_test)

    
if __name__=='__main__':
    print('hello world')
#    assert 2==3, '2 must equal to 3'
    
    
    
    
#    cancer = load_breast_cancer()
#    print('cancer is ', cancer.data.shape)
#    print('cancer is ', len(cancer.data))
#    
#    rng = np.random.RandomState(42)
#    noise = rng.normal(size=(len(cancer.data), 50))
#    print('noise.shape is ', noise.shape)
#    
#    X_w_noise = np.hstack([cancer.data, noise])
#    print('X_w_noise is ', X_w_noise.shape)
#    
#    X_train, X_test, y_train, y_test = train_test_split(
#            X_w_noise, cancer.target, random_state=0, test_size=.5)
#    select = SelectPercentile(percentile=50)
#    select.fit(X_train, y_train)
#    X_train_selected = select.transform(X_train)
#    
#    print(X_train.shape, y_train.shape)
#    print(X_train_selected.shape)
#    
#    mask = select.get_support()
#    print(mask.shape)
#    print(mask.reshape(1, -1).shape)
#    
#    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    
    
    
    
    
#    x_train = []
#    y_train = []
#    for x in np.linspace(-5, 5, 10001):
#        x_train.append([x])
#        y_train.append([x**2])
#        
#    x_test = []
#    y_test = []
#    for x in np.linspace(-10, 10, 20001):
#        x_test.append([x])
#        y_test.append([x**2])
#    
#    fig = plt.figure()
#    print('training started...')
#    ax = fig.add_subplot(4, 1, 1)
#    ax.plot(x_train, y_train)
#    ax.set_title('training data')

##############################################
    
#    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}   
#                  
#    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=5)
#    grid_search.fit(x_train, y_train)
#    y_test = grid_search.predict(x_test)
#    
#    print('grid_search.best_params_ is ', grid_search.best_params_)
#    print('grid_search.best_score_ is ', grid_search.best_score_)
#    ax = fig.add_subplot(4, 1, 2)
#    ax.plot(x_test, y_test)
#    ax.set_title('SVR')
    
##############################################

#    param_grid = {'n_estimators': [100, 500, 1000],
#                  'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]}
#                  
#    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
#    grid_search.fit(x_train, y_train)
#    y_test = grid_search.predict(x_test)
#    
#    print('grid_search.best_params_ is ', grid_search.best_params_)
#    print('grid_search.best_score_ is ', grid_search.best_score_)
#    ax = fig.add_subplot(4, 1, 3)
#    ax.plot(x_test, y_test)
#    ax.set_title('GBRT')

##############################################

#    param_grid = {'alpha': [0.1, 0.3, 0.7, 1.0, 2.0, 5.0]}
#                  
#    grid_search = GridSearchCV(Lasso(), param_grid, cv=5)
#    grid_search.fit(x_train, y_train)
#    y_test = grid_search.predict(x_test)
#    
#    print('grid_search.best_params_ is ', grid_search.best_params_)
#    print('grid_search.best_score_ is ', grid_search.best_score_)
#    ax = fig.add_subplot(4, 1, 4)
#    ax.plot(x_test, y_test)
#    ax.set_title('LASSO')

##############################################

#    param_grid = {'n_neighbors': [3, 5, 10, 20],
#                  'weights': ['uniform', 'distance']}
#                  
#    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
#    grid_search.fit(x_train, y_train)
#    y_test = grid_search.predict(x_test)
#    
#    print('grid_search.best_params_ is ', grid_search.best_params_)
#    print('grid_search.best_score_ is ', grid_search.best_score_)
#    ax = fig.add_subplot(4, 1, 4)
#    ax.plot(x_test, y_test)
#    ax.set_title('KNN')
    
##############################################
    
#    clf = svm.SVR()
#    clf.fit(x_train, y_train)
#    y_test = clf.predict(x_test)
#    ax = fig.add_subplot(3, 1, 2)
#    ax.plot(x_test, y_test)
#    ax.set_title('SVR')
    
##############################################
    
#    gbr = GradientBoostingRegressor(n_estimators=100, 
#                                    max_depth=4, 
#                                    learning_rate=0.07,
#                                    random_state=0)
#    gbr.fit(x_train, y_train)
#    y_test = gbr.predict(x_test)
#    ax = fig.add_subplot(3, 1, 3)
#    ax.plot(x_test, y_test)
#    ax.set_title('GBRT')
    
##############################################
    
#    knn_reg = KNeighborsRegressor(n_neighbors=3)
#    knn_reg.fit(x_train, y_train)
#    y_test = knn_reg.predict(x_test)
#    plt.plot(x_test, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    