import numpy as np
from numpy import linalg as LA
x = np.array([[1,-1,1],
            [1,1,-1],
            [1,0,1]])

beta = np.array([-0.682,0.56,1.244]).reshape((-1,1))

print(LA.norm(beta))
print("\n")

# dot = x.T.dot(beta) 
# print(dot)

# print("\n")

# e = np.exp(1/(1+dot))

# print(e)