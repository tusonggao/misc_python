import sys
import time
from datetime import datetime

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.stats import skew
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
   
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})    
   
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))
    
if __name__=='__main__':
    print('hello world 111')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
    print('hello world 222')
    sys.exit(0)
    


#    print("score: %f" % dummy_majority.score(X_test, y_test))
    
    train_data_frame = pd.read_csv('./data/train.csv')
    test_data_frame = pd.read_csv('./data/test.csv')
    print('train_data_frame shape is ', train_data_frame.shape)
    print('test_data_frame shape is ', test_data_frame.shape)
    
#    print('test_data_frame sex value_counts \n', test_data_frame['Sex'].value_counts())    
#    outcome_data_frame = pd.DataFrame(
#                         {'Survived': [1 if x==True else 0 for x in guess_outcome]}, 
#                         index=test_data_frame.index.values)
#    outcome_data_frame.to_csv('./tsg_outcome.csv', index_label='PassengerId')

#    end_time = time.time()    
#    print('time cost is ', end_time - start_time)    
#    sys.exit(0)
    
#    cols_to_drop = ['Survived']

    X_train = train_data_frame.drop(['label'], axis=1)
    y_train = train_data_frame['label']
    X_test = test_data_frame
    
#    concated_dataframe = pd.concat([features_data_frame, test_data_frame])    
#    concated_dataframe = concated_dataframe.select_dtypes(include=['float', 'int']).copy()

#    print('features_data_frame is ', list(features_data_frame.columns))
#    print('features_data_frame is ', list(features_data_frame['Fare'])[295:307])
#    print('test_dataframe.shape is ', test_data_frame.shape)


#    features_data_frame = pd.get_dummies(features_data_frame)
#    test_data_frame = pd.get_dummies(test_data_frame)
#    adjust_test_dataframe(features_data_frame, train_data_frame)
    
#    features_cols = list(set(train_data_frame.columns)-set(['Survived']))


#    print(data_dummies.dtypes)
#    col_names = list(data_dummies.columns)
#    train_cols = list(set(train_data_frame.columns)-
#                      set(['PassengerId', 'Survived', 'Name', 'Ticket']))
    
#    X_train, X_test, y_train, y_test = train_test_split(
#                                            train_data_frame[train_cols], 
#                                            train_data_frame['SalePrice'], 
#                                            random_state=42)


######################################################################
            
#    lr = LinearRegression().fit(X_train, y_train)
                                            
#    best_ratio = 0
#    best_score = -1000
#    scores_mean_list = []
#    ratio_list = []
#    for ratio in range(10, 100, 10):
#        print('ratio is ', ratio)
#        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
#        rf = RandomForestRegressor(n_estimators=1000,
#                               max_features=int(len(train_cols)*ratio/100),
#                               max_depth=4,                               
#                               n_jobs=4)
#        scores = cross_val_score(rf, train_data_frame[train_cols], 
#                                 train_data_frame['SalePrice'], 
#                                 cv=kfold)
#        print('ratio is ', ratio, 'scores mean is ', scores.mean())
#        scores_mean_list.append(scores.mean())
#        ratio_list.append(ratio)
#        if best_score < scores.mean():
#            best_score = scores.mean()
#            best_ratio = ratio

#    plt.plot(ratio_list, scores_mean_list)
#    plt.show()
    
#    rf = RandomForestRegressor(n_estimators=10000, 
#                               max_features=int(len(train_cols)*best_ratio/100),
#                               max_depth=4,
#                               n_jobs=4).fit(
#                               train_data_frame[train_cols],
#                               train_data_frame['SalePrice'])

###########################################################################

    gbc = GradientBoostingClassifier(n_estimators=1000,
                                     random_state=42,
                                     max_depth=10,
                                     learning_rate=0.08
                                     )
    print('---------- training start at ', datetime.now(), '------------')
    start_time = time.time()
    ovr_classifier = OneVsRestClassifier(gbc).fit(X_train, y_train)
    end_time = time.time()
    print('training time cost is ', end_time - start_time)
    start_time = time.time()
    outcome = ovr_classifier.predict(X_test)
    end_time = time.time()
    print('test time cost is ', end_time - start_time)

#svr grid_search.best_params_ is  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 5000}
#svr grid_search.best_score_ is  0.848484848485
#best score is  0.997755331089
#time cost is  6576.964999914169

#    param_grid = {'n_estimators': [300],
#                  'learning_rate': [0.001, 0.001, 0.1],
#                  'max_depth': [2, 4, 6, 8, 10, None]}
#                  
#    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), 
#                               param_grid, cv=5)
#    grid_search.fit(X_train, y_train)
#    test_score = grid_search.score(X_train, y_train)
#    outcome = grid_search.predict(X_test)
#    gbr_prob = grid_search.predict_proba(X_test)
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    

#    gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, 
#                                     learning_rate=0.01,
#                                     random_state=42)
#    scores = cross_val_score(gbr, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
                               
#    gbr.fit(X_train, y_train)
#    print("accuracy on training set:", gbr.score(X_train, y_train))
#    outcome = gbr.predict(X_test)
#    gbr_prob = gbr.predict_proba(X_test)
#    print('gbr_prob is ', gbr_prob)
#    print('shape of gbr_prob is ', gbr_prob.shape)
#    gbr_prob_frame = pd.DataFrame({'0': gbr_prob[:, 0], 
#                                   '1':gbr_prob[:, 1],
#                                   'outcome': outcome},
#                                   index=X_test.index.values)
#    gbr_prob_frame.to_csv('./gbr_prob_frame.csv', index_label='PassengerId')

######################################################################

#    rf = RandomForestClassifier(n_estimators=3000,
#                                n_jobs=2,
#                                random_state=42,
#                                min_samples_leaf=3
#                                )
#    print('training start...')
#    start_time = time.time()
#    ovr_classifier = OneVsRestClassifier(rf).fit(X_train, y_train)
#    end_time = time.time()
#    print('training time cost is ', end_time - start_time)
#    start_time = time.time()    
#    outcome = ovr_classifier.predict(X_test)
#    end_time = time.time()    
#    print('test time cost is ', end_time - start_time)
    

#    param_grid = {'n_estimators': [1000],
#                  'max_depth': [8, 15, 30, None],
#                  'max_features': [0.5, 'auto', None],
#                  'min_samples_leaf': [3, 4, 5, 6],
#                  'oob_score': [True]}
#    
#    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
#                               param_grid, cv=5, n_jobs=2)
#    grid_search.fit(X_train, y_train)
#    test_score = grid_search.score(X_train, y_train)
#    outcome = grid_search.predict(X_test)
#    
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    
#    rf = RandomForestClassifier(n_estimators=1000,
#                                n_jobs=2,
#                                random_state=42,
#                                min_samples_leaf=5,
#                                oob_score=True
#                                )
#    scores = cross_val_score(rf, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
                               
#    rf.fit(X_train, y_train)
#    print("accuracy on training set:", rf.oob_score_)
#
#    outcome = rf.predict(X_test)
#    rf_prob = rf.predict_proba(X_test)
#    print('shape of rf_prob is ', rf_prob.shape)
#    rf_prob_frame = pd.DataFrame({'0': rf_prob[:, 0], 
#                                  '1': rf_prob[:, 1],
#                                  'outcome': outcome},
#                                  index=X_test.index.values)
#    rf_prob_frame.to_csv('./rf_prob_frame.csv', index_label='PassengerId')
    
    
#######################################################################
    
    ### Logistic Regression 需要feature rescaling
    
#    df_concated = pd.concat([features_data_frame, test_data_frame])
#    scaler = StandardScaler().fit(df_concated)
#    features_data_frame = scaler.transform(features_data_frame)
#    test_data_frame = scaler.transform(test_data_frame)
#    
#    param_grid = {'penalty': ['l1', 'l2'],
#                  'C': [0.01, 0.1, 1, 10, 100],
#                  'solver': ['liblinear']}
#                  
#    grid_search = GridSearchCV(LogisticRegression(random_state=5), 
#                               param_grid, cv=5)
#    grid_search.fit(features_data_frame, target_data_frame)
#    test_score = grid_search.score(features_data_frame, target_data_frame)
#    outcome = list(grid_search.predict(test_data_frame))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    
    
#    lr = LogisticRegression()
#    lr.fit(features_data_frame, target_data_frame)
#    print("accuracy on training set:", lr.score(features_data_frame, 
#                                                 target_data_frame))
#
#    print('length of test_data_frame ', len(test_data_frame))
#    outcome = list(lr.predict(test_data_frame))
#    print('length of outcome is ', len(outcome))
    
#######################################################################
    
#    param_grid = {'kernel': ["rbf"],
#                  'C'     : np.logspace(-5, 5, num=11, base=10.0),
#                  'gamma' : np.logspace(-5, 5, num=11, base=10.0)}
#    df_concated = pd.concat([features_data_frame, test_data_frame])
#    scaler = StandardScaler().fit(df_concated)
#    x_train_scaled = scaler.transform(features_data_frame)
#    x_test_scaled = scaler.transform(test_data_frame)
#    
#    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
#    grid_search.fit(x_train_scaled, target_data_frame)
#    test_score = grid_search.score(x_train_scaled, target_data_frame)
#    outcome = list(grid_search.predict(x_test_scaled))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)


#######################################################################

#    sum_prob = gbr_prob + rf_prob
#    outcome = sum_prob[:, 0] < sum_prob[:, 1]
#    sum_prob_frame = pd.DataFrame({'0': sum_prob[:, 0], '1':sum_prob[:, 1],
#                                   'outcome': outcome},
#                                  index=X_test.index.values)
#    sum_prob_frame.to_csv('./sum_prob_frame.csv', index_label='PassengerId')

    outcome_data_frame = pd.DataFrame(data = outcome, 
                                      index = np.array(X_test.index.values)+1, 
                                      columns = ['Label'])
    
#    outcome_data_frame = pd.DataFrame(data = sum_prob['outcome'], 
#                                      index = X_test.index.values, 
#                                      columns = ['Survived'])   
                         
#    outcome_data_frame = pd.DataFrame(
#                         {'Survived': [1 if x==True else 0 for x in outcome]}, 
#                         index=X_test.index.values)
                         
#    outcome_data_frame = outcome_data_frame.set_index('PassengerId')
    outcome_data_frame.to_csv('./submission_tsg_new.csv', index_label='ImageId')
    
#######################################################################
