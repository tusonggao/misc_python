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
	

file_name = 'C:/github_base/misc_python/sudoku_solver/input.txt'
input_matrix = read_input(file_name)
print('input_matrix is ', input_matrix)

input_matrix_new = input_matrix.copy()
print('input_matrix_new is ', input_matrix_new)

#solve(0, 0, input_matrix)



def next_position(i, j):
    num = i*9 + j + 1
    return divmod(num, 9)

def prev_position(i, j):
    num = i*9 + j - 1
    return divmod(num, 9)

def check_ok(i, j, input_matrix):
    counter = Counter(input_matrix[i])
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            return False
    
    counter = Counter([input_matrix[x][j] for x in range(9)])
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            return False
    
    counter = Counter([input_matrix[x][y] for x in range(i//3*3, i//3*3 + 3) 
                                          for y in range(j//3*3, j//3*3 + 3)])
    for val, cnt in counter.items():
        if cnt>=2 and val!='X':
            return False
    
    return True
    
    

def solve(i, j, input_matrix):
    if FOUND==True:
        return
    
    if input_matrix_original[i][j]=='X':
        for val in range(1, 10)
    right_placement = check_ok(i, j, input_matrix)
    if i==8 and j==8 and right_placement:
        print('found one solution', input_matrix)
        FOUND = True
    elif right_placement:
        solve(i, j, input_matrix)
        
for pos in range(9*9):
    i, j = divmod(pos, 9)
    print(i, j, input_matrix[i][j])
        
        
			    