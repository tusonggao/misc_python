from collections import Counter

FOUND = False

def read_input(file_name):
    input_matrix = []
    with open(file_name) as file:
        for line in file:
            line = line.strip()
            if len(line)==0:
                continue
            list_n = [s for s in line.strip() if s!=' ']
            input_matrix.append(list_n)
    return input_matrix
	
def print_outcome(input_matrix):
    for i in range(9):
        print(' '.join([str(v) for v in input_matrix[i]]))
    print('\n---------------------------\n')

def valid_position(i, j):
    return 0 <= i <= 8 and 0 <= j <=8

def next_position(i, j):
    num = i*9 + j + 1
    return divmod(num, 9)

def prev_position(i, j):
    num = i*9 + j - 1
    return divmod(num, 9)

def check_ok(i, j, input_matrix):
    print('in checkok i, j:', i, j, 'current value is ', input_matrix[i][j])
    ll1 = input_matrix[i]
    print('ll1 is ', ll1)
    counter = Counter(ll1)
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            print('check_ok is ', False)
            return False
    
    ll2 = [input_matrix[x][j] for x in range(9)]
    print('ll2 is ', ll2)
    counter = Counter(ll2)
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            print('check_ok is ', False)
            return False
        
    ll3 = [input_matrix[x][y] for x in range(i//3*3, i//3*3 + 3) 
                              for y in range(j//3*3, j//3*3 + 3)]
    counter = Counter(ll3)
    print('ll3 is ', ll3)
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            print('check_ok is ', False)
            return False
    
    print('check_ok is ', True)
    return True


def solve(i, j, input_matrix):
    while input_matrix_original[i][j]!='X':
        i, j = next_position(i, j)
        if valid_position(i, j)==False: 
            return
    
    pos_x, pos_y = i, j
    for val in range(1, 10):
        input_matrix[pos_x][pos_y] = str(val)
        if check_ok(pos_x, pos_y, input_matrix):
            print_outcome(input_matrix)
#            name = input("please input：");
            if i==8 and j==8:
                print('ok, found one')
                print_outcome(input_matrix)
            else:
                i, j = next_position(i, j)
                solve(i, j, input_matrix)
        if input_matrix_original[pos_x][pos_y]=='X':
            input_matrix[pos_x][pos_y] = 'X'
            

file_name = 'C:/github_base/misc_python/sudoku_solver/input.txt'
input_matrix_original = read_input(file_name)
print('input_matrix is ', input_matrix_original)
print_outcome(input_matrix_original)

input_matrix_new = input_matrix_original.copy()
#print('input_matrix_new is ', input_matrix_new)

solve(0, 0, input_matrix_new)
       
			    