import numpy as np
np.random.seed(2017)

#.reshape(w.shape)



w = np.random.randint(20, 120, size=(30))
print(w)
sign = np.where(np.random.randint(0, 2, size=w.shape)==0, -1, 1)
w *= sign
b = 10

print(sign)
print(w)

#.reshape(y.shape)

output_string = ''
output_string += '30 20\n'

results_string = 'w is: '
for w_v in w:
    results_string += str(w_v) + ' '
results_string += '\n'
    

train_num = 20

for i in range(train_num):
    x = np.random.randint(10, 200, size=(30))
    sign = np.where(np.random.randint(0, 2, size=w.shape)==0, -1, 1)
    x *= sign
    y = np.sum(w*x) + b
    y += (1.5*np.random.randn())
#    print('y is ', y)
    for x_v in x:
        output_string += str(x_v) + ' '
    output_string += str(y) + '\n'

output_string += '5 \n'
for i in range(5):
    x = np.random.randint(10, 200, size=(30))
    sign = np.where(np.random.randint(0, 2, size=w.shape)==0, -1, 1)
    x *= sign
    y = np.sum(w*x)
    print('y is ', y)
    for x_v in x:
        output_string += str(x_v) + ' '
        results_string += str(x_v) + ' '
    output_string += '\n'
    results_string += str(y) + '\n'
    

with open('test_data_generated.txt', 'w') as test_file:
    test_file.write(output_string)
    
with open('answer_data_generated.txt', 'w') as test_file:
    test_file.write(results_string)
    
    
        
