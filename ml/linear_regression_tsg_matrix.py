import numpy as np
import time

class Linear_Regression_Classifier():
    def __init__(self, learning_rate=0.005, delta=10**(-7), max_loop_num=10**6):
        self.learning_rate = learning_rate  # learning rate
        self.delta = delta
        self.max_loop_num = max_loop_num

    def fit(self, X_train, y_train):
        self.m = X_train.shape[0]   # 训练样本个数
        self.n = X_train.shape[1]   # 训练样本feature个数
        
        self.w = np.random.rand(self.n, 1)  # parameter
#        self.w = np.zeros(shape=(self.n, 1))  # parameter
        self.X_train = X_train
        self.y_train = y_train
        
        self.running_loop_count = 0
        while True:
            self.running_loop_count += 1
            step = self.update_coefficient()
            if step < self.delta:
                break
            if self.running_loop_count > self.max_loop_num:
                break
    
    def update_coefficient(self):
        y_bias = np.dot(self.X_train, self.w) - self.y_train

        step = np.dot(self.X_train.T, y_bias)
        step *=(2./self.m)*self.learning_rate
        self.w -= step
        step_max = np.max(np.abs(step))
        return step_max
    
    def current_loss(self):
        y_hat = np.dot(self.X_train, self.w)
        return np.sum((y_hat - self.y_train)**2)
    
    def predict(self, X_test):
        return np.dot(X_test, self.w)
   
    
def take_input_value():
#    file_in = open('in_data.txt')
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
    regr = Linear_Regression_Classifier()
    regr.fit(X_train, y_train)
    y_test = regr.predict(X_test)
    end_t = time.time()
    
    print('y_test is {} cost time: {} sec '.format(y_test, end_t-start_t))
    print('parameter is ', regr.w)
    print('count is ', regr.running_loop_count)



