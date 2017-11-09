import time

#m = 10  # 训练集中包含instance 的个数
#n = 10  # 每个instance的feature的个数

#w_list = [0. for _ in range(n+1)]   # (n+1)个数值 最后一个数值为b

alpha = 0.005     #  learning rate
delta = 10**(-7)

#x_instances = []   # m X n+1 矩阵   最后一列数值为1 对应 b
#y = [] # m X 1 矩阵
#
#test_num = 0
#x_instances_to_be_test = []
     
def compute_F_t(t):
    global w_list, b, x_instances, y
    sum_val = -y[t]
    for i in range(n+1):
        sum_val += w_list[i]*x_instances[t][i]
    return sum_val

def update_coefficient():
    global m, n, w_list, b, alpha, x_instances, y
    
    F_list = []
    for i in range(m):
        F_list.append(compute_F_t(i))
    
    step_max = -10**9
    for i in range(n+1):
        derivative = 0.
        for j in range(m):
            derivative +=  F_list[j] * x_instances[j][i]
        derivative *= 2./m
        step = alpha*derivative
        w_list[i] -= step
        if step_max < abs(step):
            step_max = abs(step)    
    return step_max
  
def main():
    global delta
    count = 1
    while True:
        count += 1
        step = update_coefficient()
        if step < delta:
            break
#    print('count: {0} step: {1}'.format(count, step))

        
def take_input_value():
    file_in = open('in_data.txt')
#    global m, n, x_instances, y, test_num, x_instances_to_be_test
    n, m = map(int, file_in.readline().split())
    x_instances, y = [], []
    for i in range(m):
        x_instance = []
        x_instance = list(map(float, file_in.readline().split()))
        y.append(x_instance[-1])
        x_instance[-1] = 1.0
        x_instances.append(x_instance)
    test_num = int(file_in.readline())
    x_instances_to_be_test = []
    for i in range(test_num):
        x_instance = list(map(float, file_in.readline().split()))
        x_instance.append(1.0)
        x_instances_to_be_test.append(x_instance)
    w_list = [0. for _ in range(n+1)] 
    return m, n, x_instances, y, test_num, x_instances_to_be_test, w_list

def compute_test_data():
    global test_num, x_instances_to_be_test, w_list
    computed_prices = []
    for i in range(test_num):
        price_val = 0.
        for j in range(n+1):
            price_val += w_list[j]*x_instances_to_be_test[i][j]
        computed_prices.append(price_val)
    return computed_prices

if __name__ == '__main__':
    m, n, x_instances, y, test_num, x_instances_to_be_test, w_list = take_input_value()
    
    start_t = time.time()
    main()
    y_test = compute_test_data()
    end_t = time.time()
    print('y_test is {} cost time: {} sec '.format(y_test, end_t-start_t))
    print('parameter is ', w_list)



