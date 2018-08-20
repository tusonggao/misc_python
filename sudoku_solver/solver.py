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

FOUND = False

def next_position(i, j):
    num = i*9 + j + 1
    return divmod(num, 9)

def prev_position(i, j):
    num = i*9 + j - 1
    return divmod(num, 9)


def solve(i, j, input_matrix):
    if Found:
        return
    right_placement = check(i, j, input_matrix)
    if i==8 and j==8 and right_placement:
        print('found one solution', input_matrix)
        Found = True
    elif right_placement:
        solve(i, j, input_matrix)
        
#for pos in range(9*9):
#    i, j = divmod(pos, 9)
#    print(i, j, input_matrix[i][j])
        
        
			    