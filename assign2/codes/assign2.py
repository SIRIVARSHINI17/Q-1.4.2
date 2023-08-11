import numpy as np

# Coefficient matrix of the system of equations
# equations obtained from previous questions
coefficient_matrix = np.array([[5., -7.],
                         [4., 4.]])
# Constant vector
constant_matrix = np.array([-25. ,-16.])

# Create the augmented matrix
augmented_matrix = np.column_stack((coefficient_matrix, constant_matrix))

# Use Gaussian elimination to transform the augmented matrix into row-echelon form
def gaussian_elimination(matrix):
    num_rows, num_cols = matrix.shape
    for i in range(num_rows):
        pivot_row = matrix[i]
        pivot_element = pivot_row[i]
        
        # Make the pivot element 1
        matrix[i] = pivot_row / pivot_element
        
        # Eliminate other rows
        for j in range(i + 1, num_rows):
            matrix[j] -= matrix[i] * matrix[j, i]
    return matrix

echelon_matrix = gaussian_elimination(augmented_matrix.copy())

# Back-substitution to solve for variables
def back_substitution(matrix):
    num_rows, num_cols = matrix.shape
    solution = np.zeros(num_rows)
    for i in range(num_rows - 1, -1, -1):
        solution[i] = matrix[i, -1] - np.dot(matrix[i, i+1:num_cols-1], solution[i+1:])
    return solution

solution = back_substitution(echelon_matrix)
print("Circumcenter(O):", solution)
