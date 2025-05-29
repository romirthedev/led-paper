import numpy as np
from .lattice_boltzmann_case2 import run_lb_case2

class CASE2EQN:
    def __init__(self, a_max=1.0, N=200, t_max=4.0, dt=0.002):
        self.a_max = a_max
        self.N = N
        self.t_max = t_max
        self.dt = dt
        self.da = a_max / N
        self.a = np.linspace(0, a_max, N, endpoint=False)
        self.timesteps = int(t_max / dt)
        self.f0 = 50 * np.exp(-((self.a-0.2)**2) / 0.0005)
        self.G = 0.434 + 0.2604 * self.a

    def simulate_upwind(self):
        f = self.f0.copy()
        f_all = np.zeros((self.timesteps, self.N))
        f_all[0] = f
        for n in range(1, self.timesteps):
            f_new = f.copy()
            Gf = self.G * f
            # Conservative upwind for d/da(Gf)
            f_new[1:] = f[1:] - self.dt / self.da * (Gf[1:] - Gf[:-1])
            # Left boundary: Dirichlet ghost node = 0
            f_new[0] = f[0] - self.dt / self.da * (Gf[0] - 0)
            # Right boundary: no-flux (Neumann)
            f_new[-1] = f_new[-2]
            f = f_new
            f_all[n] = f
        return f_all

    def simulate_led(self):
        f_all, t_vec = run_lb_case2(self.f0, self.a, self.G, self.t_max, self.dt, self.N, self.a_max)
        return f_all, t_vec

    def get_data(self):
        return self.f0, self.a, np.arange(self.timesteps) * self.dt, self.G 