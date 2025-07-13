import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g, L = 32.17, 2

def f(t, y):
    return [y[1], 
            - (g / L) * np.sin(y[0])]

t_eval = np.linspace(0, 2, 100)
IC=[np.pi/6, 0]

sol = solve_ivp(f, [0, 2], IC, t_eval=t_eval)

print(f"theta: {sol.y[0]}")

plt.plot(sol.t, sol.y[0])
plt.xlabel('t'), plt.ylabel('theta(t)'), plt.grid()
plt.show()
