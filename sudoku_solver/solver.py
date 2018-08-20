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

solve(0, 0, input_matrix)


def next_position(i, j):
    num = i*9 + j + 1
    return divmod(num, 9)


def solve(i, j, input_matrix):
    right_placement = check(i, j, input_matrix)
    if i==8 and j==8 and right_placement:
        print('found one solution', input_matrix)
    else if right_placement:
        solve()
        
for pos in range(9*9):
    i, j = divmod(pos, 9)
    print(i, j, input_matrix[i][j])
    for val in range(1, 10):
        
			    