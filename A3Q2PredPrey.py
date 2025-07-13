import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lotka-Volterra predator-prey system
def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1 * x + 0.02 * x * y
    dydt = 0.2 * y - 0.025 * x * y
    return [dxdt, dydt]

# Initial conditions
x0 = 6
y0 = 6
t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)

# Solve the system
soln = solve_ivp(lotka_volterra, t_span, [x0, y0], t_eval=t_eval)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t_eval, soln.y[0], label='x(t) - Predators')
plt.plot(t_eval, soln.y[1], label='y(t) - Prey')
plt.xlabel('Time t')
plt.ylabel('Population (thousands)')
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.grid()
plt.show()


