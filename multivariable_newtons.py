import numpy as np  # NumPy library for matrix operations

# Define the system of nonlinear equations
def F(X):
    x, y = X  # unpack variables from vector
    # return function values as a vector
    return np.array([x**2 + y**2 - 4,   # f1(x,y)
                     x - y])            # f2(x,y)

# Define the Jacobian matrix
def J(X):
    x, y = X  # unpack variables
    # return Jacobian matrix
    return np.array([[2*x, 2*y],  # ∂f1/∂x , ∂f1/∂y
                     [1, -1]])    # ∂f2/∂x , ∂f2/∂y

# Initial guess (starting point)
X = np.array([1.0, 1.0])

# Perform iterations
for i in range(5):
    
    # Solve J * delta = -F using Gaussian elimination
    delta = np.linalg.solve(J(X), -F(X))
    
    # Update the solution
    X = X + delta
    
    # Print current iteration result
    print("Iteration", i+1, ":", X)
