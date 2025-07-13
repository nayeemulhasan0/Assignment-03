


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Problem 1: y' = te^(3t) - 2y, y(0) = 0
def f1(t, y):
    return t * np.exp(3*t) - 2*y

def exact1(t):
    return (1/5) * t * np.exp(3*t) - (1/25) * np.exp(3*t) + (1/25) * np.exp(-2*t)

# Problem 2: y' = 1 + (t-y)^2, y(2) = 1
def f2(t, y):
    return 1 + (t - y)**2

def exact2(t):
    return t + 1/(1 - t)

# Solve Problem 1
t1 = np.linspace(0, 1, 100)
# odeint
sol1_odeint = odeint(lambda y, t: f1(t, y), 0, t1)
# solve_ivp
sol1_ivp = solve_ivp(f1, [0, 1], [0], t_eval=t1, method='RK45')

# Solve Problem 2
t2 = np.linspace(2, 3, 100)
# odeint
sol2_odeint = odeint(lambda y, t: f2(t, y), 1, t2)
# solve_ivp
sol2_ivp = solve_ivp(f2, [2, 3], [1], t_eval=t2, method='RK45')

# Calculate exact solutions
exact_sol1 = exact1(t1)
exact_sol2 = exact2(t2)

# Calculate errors
error1_odeint = np.abs(sol1_odeint.flatten() - exact_sol1)
error1_ivp = np.abs(sol1_ivp.y[0] - exact_sol1)
error2_odeint = np.abs(sol2_odeint.flatten() - exact_sol2)
error2_ivp = np.abs(sol2_ivp.y[0] - exact_sol2)

# Plotting
fig, ((sub1, sub2), (sub3, sub4)) = plt.subplots(2, 2, figsize=(12, 8))

# Problem 1 - Solutions
sub1.plot(t1, exact_sol1, 'k-', linewidth=2, label='Exact')
sub1.plot(t1, sol1_odeint, 'r--', label='odeint')
sub1.plot(t1, sol1_ivp.y[0], 'b:', label='solve_ivp')
sub1.set_title("Problem 1: y' = te^(3t) - 2y")
sub1.set_xlabel('t')
sub1.set_ylabel('y')
sub1.legend()
sub1.grid()

# Problem 1 - Errors
sub2.semilogy(t1, error1_odeint, 'r-', label='odeint error')
sub2.semilogy(t1, error1_ivp, 'b-', label='solve_ivp error')
sub2.set_title('Problem 1: Absolute Errors')
sub2.set_xlabel('t')
sub2.set_ylabel('|Error|')
sub2.legend()
sub2.grid()

# Problem 2 - Solutions
sub3.plot(t2, exact_sol2, 'k-', linewidth=2, label='Exact')
sub3.plot(t2, sol2_odeint, 'r--', label='odeint')
sub3.plot(t2, sol2_ivp.y[0], 'b:', label='solve_ivp')
sub3.set_title("Problem 2: y' = 1 + (t-y)²")
sub3.set_xlabel('t')
sub3.set_ylabel('y')
sub3.legend()
sub3.grid()

# Problem 2 - Errors
sub4.semilogy(t2, error2_odeint, 'r-', label='odeint error')
sub4.semilogy(t2, error2_ivp, 'b-', label='solve_ivp error')
sub4.set_title('Problem 2: Absolute Errors')
sub4.set_xlabel('t')
sub4.set_ylabel('|Error|')
sub4.legend()
sub4.grid()

plt.show()
