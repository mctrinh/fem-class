# Import required packages
import numpy as np
import matplotlib.pyplot as plt


# Generate data
x = np.linspace(0,1,100)

# Rayleigh-Ritz Method: 1 Parameter (N=1)
u1 = -0.1667 * x * (1 - x)

# Rayleigh-Ritz Method: 2 Parameters (N=2)
u2 = -0.0813 * x * (1 - x) - 0.1707 * x**2 * (1-x)

# Exact solution
u_exact = (np.sin(x) + 2 * np.sin(1-x))/np.sin(1) + x**2 - 2

# Create figure
fig = plt.figure()

# Plot and show our data
plt.plot(x, u1, '-g', label='1-parameter R-R')
plt.plot(x, u2, '-b', label='2-parameter R-R')
plt.plot(x, u_exact, '-r', label='Exact')
plt.title("Dirichlet Problem")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()