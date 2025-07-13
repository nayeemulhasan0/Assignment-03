import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of ODEs
def competition_model(t, z):
    x, y = z
    dxdt = x * (2 - 0.4*x - 0.3*y)
    dydt = y * (1 - 0.1*y - 0.3*x)
    return [dxdt, dydt]

# Time span and evaluation points
t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)

# Initial conditions for the four cases
IC= [
    [1.5, 3.5],
    [1, 1],
    [2, 7],
    [4.5, 0.5]
]

# Plot results
fig, ((sub1, sub2), (sub3, sub4)) = plt.subplots(2, 2, figsize=(12, 8))
# for i, (x0, y0) in enumerate(initial_conditions, 1):


sol = solve_ivp(competition_model, t_span, IC[0], t_eval=t_eval)
sub1.plot(sol.t, sol.y[0],label="x")
sub1.plot(sol.t, sol.y[1],label="y")
sub1.legend()
sub1.grid()


sol = solve_ivp(competition_model, t_span, IC[1], t_eval=t_eval)
sub2.plot(sol.t, sol.y[0],label="x")
sub2.plot(sol.t, sol.y[1],label="y")
sub2.legend()
sub2.grid()


sol = solve_ivp(competition_model, t_span, IC[2], t_eval=t_eval)
sub3.plot(sol.t, sol.y[0],label="x")
sub3.plot(sol.t, sol.y[1],label="y")
sub3.legend()
sub3.grid()


sol = solve_ivp(competition_model, t_span, IC[3], t_eval=t_eval)
sub4.plot(sol.t, sol.y[0],label="x")
sub4.plot(sol.t, sol.y[1],label="y")
sub4.legend()
sub4.grid()



plt.show()
