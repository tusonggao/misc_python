import numpy as np
import matplotlib.pyplot as plt
import time

import xgboost


points  = np.array([[2, 1],
                    [3, 1],
                    [4, 1],
                    [3, 2],
                    [4, 3],
                    [100, 100]])

num_points = len(points)

print('num_points is ', num_points)

X = points[:, :-1]  # n dimension, m instances, so X is m>n matrix
y = points[:, -1]   # m x 1 vector
print('x is ', X)
print('y is ', y)

w = 0   # 1xn vector
b = 0
print('np.sum(w*x+b-y) is ', np.sum(w*x+b-y))

current_parameter = {'w': 0.0, 'b': 0.0}

#learning_rate = 2.0
learning_rate = 0.0002

begin_t = time.time()

count = 0
while True:
    count += 1
#    if count > 20:
#        break
#    print('round is ', count)
    w = current_parameter['w']
    b = current_parameter['b']
#    print('w is ', w, 'b is ', b)
    step_w = 2./num_points*np.sum(np.dot((w*x+b-y), x))
    step_b = 2./num_points*np.sum(w*x+b-y)
    current_parameter['w'] = w - learning_rate*step_w
    current_parameter['b'] = b - learning_rate*step_b
    delta = max(abs(step_w), abs(step_b))
#    print('delta is ', delta)
    if delta < 10**(-3):
        break
end_t = time.time()
print('cost time is ', end_t-begin_t)

















