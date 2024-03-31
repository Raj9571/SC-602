import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def system_dynamics(t, x, k):
    x1, x2 = x
    u = -k * (x1 + x2) - x1**2 * x2
    dx1_dt = -x1 + x1**2 * x2
    dx2_dt = -x1**3 - x2 + u
    return dx1_dt, dx2_dt, u

def solve_system(t_span, initial_conditions, k):
    def rhs(t, x):
        dx1_dt, dx2_dt, _ = system_dynamics(t, x, k)
        return [dx1_dt, dx2_dt]
    
    sol = solve_ivp(rhs, t_span, initial_conditions, t_eval=np.linspace(t_span[0], t_span[1], 300))
    u_vals = [system_dynamics(t, y[:2], k)[2] for t, y in zip(sol.t, sol.y.T)]
    return sol, u_vals

# Initial conditions, time span, and control gain
initial_conditions = [1, -1]
t_span = [0, 10]
k = 5 #control gain

sol, u_vals = solve_system(t_span, initial_conditions, k)

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='$x_1(t)$')
plt.plot(sol.t, sol.y[1], label='$x_2(t)$')
plt.title('States $x_1$ and $x_2$ Over Time')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(sol.t, u_vals, label='Control Input $u(t)$', color='red')
plt.title('Control Input $u$ Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Control Input $u$')
plt.legend()
plt.grid(True)
plt.show()
