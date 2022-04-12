# Import required packages
import numpy as np
import matplotlib.pyplot as plt


# Generate data
x = np.linspace(0,1,100)

# Rayleigh-Ritz Method: 1 Parameter (N=1)
u1 = 1.125 * x

# Rayleigh-Ritz Method: 2 Parameters (N=2)
u2 = 1.295 * x - 0.15108 * x**2

# Exact solution
u_exact = (2 * np.cos(1-x) - np.sin(x)) / np.cos(1) + x**2 - 2

# Create figure
fig = plt.figure()

# Plot and show our data
plt.plot(x, u1, '-g', label='1-parameter R-R')
plt.plot(x, u2, '-b', label='2-parameter R-R')
plt.plot(x, u_exact, '-r', label='Exact')
plt.title("Mixed BVP")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()