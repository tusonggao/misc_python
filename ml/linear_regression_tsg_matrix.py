import numpy as np
import pandas as pd

m = 10  # 训练集中包含instance 的个数
n = 10  # 每个instance的feature的个数

w = []   # (n+1) X 1 向量

alpha = 0.005       #  learning rate
delta = 10**(-7)

X_train = []   # m X n+1 矩阵   最后一列数值为1 对应 b
y_train = []   # m X 1 矩阵

test_num = 0
X_train = []   # m X n+1 矩阵   最后一列数值为1 对应 b
y_train = []   # m X 1 矩阵

#    for i in range(n+1):
#        derivative = y_hat*X_train[:, i]
#        derivative *= 2./m
#        step = alpha*derivative
#        w -= step
#        if step_max < abs(step):
#            step_max = abs(step)

class Linear_Regression_Classifier():
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        self.m = X_train.shape[0]   # 训练样本个数
        self.n = X_train.shape[1]   # 训练样本feature个数
        
        self.w = np.random.rand(n+1, w)
    
    def test(self, X_test):
        return np.dot(X_test, w)

def linear_regression_fit()


def update_coefficient():
    global m, n, w, alpha, X_train, y_train
    
    print('pre X_train shape is ', X_train.shape, 'w shape is', w.shape)
    y_hat = np.dot(X_train, w)
    y_bias = np.dot(X_train, w) - y_train
    print('y_hat shape is ', y_hat.shape, 
          'y_bias shape is ', y_bias.shape, 
          'y_train shape is ', y_train.shape)
    
    X_train_T = X_train.T.copy()
    step = np.dot(X_train_T, y_bias)
    step *=(2./m)*alpha
    w -= step
    step_max = np.max(np.abs(step))
    return step_max

def current_loss():
    global X_train, w, y_train
    y_hat = np.dot(X_train, w)
    return np.sum((y_hat - y_train)**2)/X_train.shape[0]
    
    
def main():
    global delta
    count = 1
    loss_new = 10**8
    while True:
        count += 1
        step = update_coefficient()
        loss_old = loss_new
        loss_new = current_loss()
        print('current loss_new is ', loss_new, 'loss_old is ', loss_old, 'w is ', w)
#        if abs(loss_old - loss_new) < delta:
#            break
        if step < delta:
            break
        if count >=10**5:
            break
        print('count: {0} step is: {1}'.format(count, step))

        
def take_input_value():
    global m, n, test_num, X_train, w, y_train, X_test
    
    file_in = open('in_data.txt')
    x_instances, y_list = [], []
    n, m = map(int, file_in.readline().split())
    for i in range(m):
        x_instance = []
        x_instance = list(map(float, file_in.readline().split()))
        y_list.append(x_instance[-1])
        x_instance[-1] = 1.0
        x_instances.append(x_instance)
    print('x_instances is ', x_instances)
    test_num = int(file_in.readline())
    print('test_num is ', test_num)
    x_instances_to_be_test = []
    for i in range(test_num):
        x_instance = list(map(float, file_in.readline().split()))
        x_instance.append(1.0)
        x_instances_to_be_test.append(x_instance)
    print('x_instances_to_be_test is ', x_instances_to_be_test)
    
    w = np.random.rand(n+1, 1)
    print('origin w is ', w)
    X_train = np.array(x_instances)
    y_train = np.array(y_list).reshape(m, 1)
    X_test = np.array(x_instances_to_be_test)


    
def compute_test_data():
    global w, X_test, y_test
    y_test = np.dot(X_test, w)
    print('y_test is ', y_test)
    return y_test

    
if __name__ == '__main__':
    take_input_value()
    print('m is n is', m, n)
    main()
    compute_test_data()



