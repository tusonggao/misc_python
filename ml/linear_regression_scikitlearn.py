import numpy as np
import time

from sklearn.linear_model import LinearRegression


def take_input_value():
    file_in = open('test_data_generated.txt')
    x_instances, y_list = [], []
    n, m = map(int, file_in.readline().split())
    for i in range(m):
        x_instance = []
        x_instance = list(map(float, file_in.readline().split()))
        y_list.append(x_instance[-1])
        x_instance[-1] = 1.0
        x_instances.append(x_instance)
    test_num = int(file_in.readline())
    x_instances_to_be_test = []
    for i in range(test_num):
        x_instance = list(map(float, file_in.readline().split()))
        x_instance.append(1.0)
        x_instances_to_be_test.append(x_instance)
    X_train = np.array(x_instances)
    y_train = np.array(y_list).reshape(m, 1)
    X_test = np.array(x_instances_to_be_test)
    
    return X_train, y_train, X_test
    
if __name__ == '__main__':
    X_train, y_train, X_test = take_input_value() # 得到输入训练数据和测试数据，每一行为一个训练样本
    print('X_train, y_train, X_test shape is ', X_train.shape, 
          y_train.shape, X_test.shape)
    
    start_t = time.time()
    regr = LinearRegression().fit(X_train, y_train)
    regr.fit(X_train, y_train)
    y_test = regr.predict(X_test)
    end_t = time.time()
    
    print('y_test is {} cost time: {} sec '.format(y_test, end_t-start_t))
    print('Coefficients: \n', regr.coef_)
    print('intercept is: \n', regr.intercept_)