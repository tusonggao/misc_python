def read_input(file_name):
    input_matrix = []
    with open(file_name) as file:
	    for line in file:
		    list_n = [s in line.strip()]
			input_matrix.append(list_n)
    return input_matrix
	
file_name = ''
input_matrix = read_input()
print('sss is ', outcome)
			    