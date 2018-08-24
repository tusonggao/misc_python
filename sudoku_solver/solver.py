from collections import Counter
import copy
import itertools

def read_input(file_name):
    local_input_matrix = []
    with open(file_name) as file:
        for line in file:
            line = line.strip()
            if len(line)==0:
                continue
            list_n = [s for s in line.strip() if s!=' ']
            local_input_matrix.append(list_n)
    return local_input_matrix
	
def print_outcome(input_matrix):
    for i in range(9):
        print(' '.join([str(v) for v in input_matrix[i]]))
    print('\n---------------------------\n')

FOUND = False

file_name = 'C:/github_base/misc_python/sudoku_solver/input2.txt'
input_matrix_original = read_input(file_name)
print('input_matrix is ', input_matrix_original)
print_outcome(input_matrix_original)

input_matrix = copy.deepcopy(input_matrix_original)
print_outcome(input_matrix)


X_NUM = list(itertools.chain(*input_matrix_original)).count('X')
print('x_num is ', X_NUM)


def valid_position(i, j):
    return 0 <= i <= 8 and 0 <= j <=8

def next_position(i, j):
    num = i*9 + j + 1
    return divmod(num, 9)

def prev_position(i, j):
    num = i*9 + j - 1
    return divmod(num, 9)

def check_ok(i, j, input_matrix):
#    print('in checkok i, j:', i, j, 'current value is ', input_matrix[i][j])
    ll1 = input_matrix[i]
    counter = Counter(ll1)
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
#            print('ll1 is ', ll1)
#            print('check_ok is ', False)
            return False
    
    ll2 = [input_matrix[x][j] for x in range(9)]
    counter = Counter(ll2)
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
#            print('ll2 is ', ll2)
#            print('check_ok is ', False)
            return False
        
    ll3 = [input_matrix[x][y] for x in range(i//3*3, i//3*3 + 3) 
                              for y in range(j//3*3, j//3*3 + 3)]
    counter = Counter(ll3)
    
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
#            print('ll3 is ', ll3)
#            print('check_ok is ', False)
            return False
    
#    print('check_ok is ', True)
    return True


def solve(pos_x, pos_y, input_matrix):
    if pos_y>=7:
        print('get here1 pos_x pos_y is ', pos_x, pos_y)
    
    global FOUND, input_matrix_original, X_NUM
    
    if FOUND==True:
#        print('already found one, i will exit')
        return
    if valid_position(pos_x, pos_y)==False:
#        print('not valid, i will exit')
        return
    
#    print('get here2 pos_x pos_y is ', pos_x, pos_y)
    
    while input_matrix_original[pos_x][pos_y]!='X':
        pos_x, pos_y = next_position(pos_x, pos_y)
#        print('get new pos_x pos_y is ', pos_x, pos_y)
        if valid_position(pos_x, pos_y)==False:
#            print('new position not valid, i will exit')
            return
    
#    print('get here3 pos_x pos_y is ', pos_x, pos_y)
    
    next_x, next_y = next_position(pos_x, pos_x)
    
    
#    print('next_x next_y is ', next_x, next_y)
    for val in range(1, 10):
        input_matrix[pos_x][pos_y] = str(val)
        if check_ok(pos_x, pos_y, input_matrix):
            X_NUM -= 1
            print('check ok:', pos_x, pos_y, input_matrix[pos_x][pos_y])
#            print_outcome(input_matrix)
#            name = input("please input：");
            if X_NUM!=0:
                solve(next_x, next_y, input_matrix)
            else:
                print('ok, found one')
                print_outcome(input_matrix)
                FOUND = True
                return
            X_NUM += 1
    if input_matrix_original[pos_x][pos_y]=='X':
        input_matrix[pos_x][pos_y] = 'X'
#    X_NUM += 1

#pos_x, pos_y = 0, 0
#pos_x, pos_y = next_position(pos_x, pos_y)
#print(pos_x, pos_y)

solve(0, 0, input_matrix)
       
			    