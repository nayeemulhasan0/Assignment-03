
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# euler based lin shoot
def shoot(h):
    x = np.arange(0, 1 + h, h)
    def euler(f, y0):
        Y = [y0]
        for xi in x[:-1]:
            Y.append(Y[-1] + h * np.array(f(xi, Y[-1])))
        return np.array(Y)
    ode = lambda x, y: [y[1], 100*y[0]]
    Y1 = euler(ode, [1, 0])
    Y2 = euler(ode, [0, 1])
    c = (np.exp(-10) - Y1[-1,0]) / Y2[-1,0]
    return x, Y1[:,0] + c*Y2[:,0]

# solve_bvp solution
fun = lambda x, y: np.vstack((y[1], 100*y[0]))
bc = lambda ya, yb: [ya[0]-1, yb[0]-np.exp(-10)]
xm = np.linspace(0, 1, 100)
yg = np.zeros((2, 100))
# bvp = solve_bvp(fun, bc, xm, yg)

# Plot
for h in [0.1, 0.05]:
    x, y = shoot(h)
    plt.plot(x, y, '--', label=f'h={h}')
# plt.plot(xm, bvp.y[0], 'k', label='solve_bvp')
plt.plot(xm, np.exp(-10*xm), 'r', label='Exact')
plt.legend(); plt.grid(True)
plt.title('Shooting Method vs solve_bvp')
plt.xlabel('x'); plt.ylabel('y')
plt.show()
