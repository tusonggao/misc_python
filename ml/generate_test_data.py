import numpy as np
np.random.seed(2017)


test_file_name = 'test_data_generated.txt'
result_file_name = 'answer_data_generated.txt'

def generate_data(train_num, feature_num, test_num):
    w = np.random.randint(20, 120, size=(feature_num)).astype('float32')
    sign = np.where(np.random.randint(0, 2, size=w.shape)==0, -1.0, 1.0)
    w *= sign
    b = np.random.randint(-10, 10)

    results_string = 'w is: '
    for w_v in w:
        results_string += str(w_v) + ' '
    results_string += '\nb is: {}\n'.format(b)

    output_string = ''
    output_string += '{} {}\n'.format(feature_num, train_num)

    for i in range(train_num):
#        x = np.random.randint(10, 200, size=(feature_num)).astype('float32')
        x= np.random.random(size=feature_num)
        sign = np.where(np.random.randint(0, 2, size=x.shape)==0, 1., 1.)
        x *= sign
        y = np.sum(w*x) + b
        y += (0.2*np.random.randn())
        for x_v in x:
            output_string += str(x_v) + ' '
        output_string += str(y) + '\n'

    output_string += '{}\n'.format(test_num)
    for i in range(test_num):
        x= np.random.random(size=feature_num)
        sign = np.where(np.random.randint(0, 2, size=x.shape)==0, 1., 1.)
#        x = np.random.randint(10, 200, size=(feature_num)).astype('float32')
#        sign = np.where(np.random.randint(0, 2, size=x.shape)==0, -1., 1.)
        x *= sign
        y = np.sum(w*x) + b
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
        
if __name__=='__main__':
    generate_data(50, 20, 5)
    
    
        
