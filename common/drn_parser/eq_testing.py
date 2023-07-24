from sympy import symbols, Eq, solve

# Define the variables
x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9 = symbols('x_0 x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9')

# Define the system of equations
equations = (Eq(x_0, 1),
             Eq(x_1, 1.0*x_2),
             Eq(x_2, 1.0*x_3),
             Eq(x_3, 1.0*x_4),
             Eq(x_4, 1.0*x_5),
             Eq(x_5, 1.0*x_6),
             Eq(x_6, 1.0*x_7),
             Eq(x_7, 1.0*x_8),
             Eq(x_8, 1.0*x_9),
             Eq(x_9, 1.0*x_9))

# Solve the system of equations
solution = solve(equations)

# Print the solution
for key in solution:
    print(f"{key} = {solution[key]}")
