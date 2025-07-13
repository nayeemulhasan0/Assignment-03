import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def syst_ode(t, y):
    x1, x1p, x1pp, x2, x2p, x2pp = y
    return [
        x1p,
        x1pp,
        -2*x2p**2 + x2,
        x2p,
        x2pp,
        -x1pp**3 + x2p + x1 + np.sin(t)
    ]

y0 = [0, 0, 0, 0, 0, 0]
t = np.linspace(0, 10, 500)
sol = solve_ivp(syst_ode, [0, 10], y0, t_eval=t)

plt.plot(sol.t, sol.y[0], label='x1')
plt.plot(sol.t, sol.y[3], label='x2')
plt.legend(); plt.grid(); plt.show()
