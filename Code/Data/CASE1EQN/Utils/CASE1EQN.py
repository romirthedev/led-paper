import numpy as np

class CASE1EQN:
    def __init__(self, a_max=1.0, N=200, t_max=4.0, dt=0.002, v=0.1):
        self.a_max = a_max
        self.N = N
        self.t_max = t_max
        self.dt = dt
        self.v = v
        self.da = a_max / N
        self.a = np.linspace(0, a_max, N, endpoint=False)
        self.timesteps = int(t_max / dt)
        self.f0 = 100 * np.exp(-self.a / 0.01)

    def simulate_upwind(self):
        f = self.f0.copy()
        f_all = np.zeros((self.timesteps, self.N))
        f_all[0] = f
        for n in range(1, self.timesteps):
            f_new = f.copy()
            f_new[1:] = f[1:] - self.v * self.dt / self.da * (f[1:] - f[:-1])
            f_new[0] = f[0] - self.v * self.dt / self.da * (f[0] - 0)
            f_new[-1] = f_new[-2]
            f = f_new
            f_all[n] = f
        return f_all

    def simulate_lax_wendroff(self):
        f = self.f0.copy()
        f_all = np.zeros((self.timesteps, self.N))
        f_all[0] = f
        c = self.v * self.dt / self.da
        for n in range(1, self.timesteps):
            f_new = f.copy()
            # Lax-Wendroff scheme (standard for advection)
            f_new[1:-1] = (f[1:-1] - 0.5 * c * (f[2:] - f[:-2])
                           + 0.5 * c**2 * (f[2:] - 2*f[1:-1] + f[:-2]))
            # Left boundary: Dirichlet ghost node = 0
            f_new[0] = f[0] - c * (f[1] - 0) + 0.5 * c**2 * (f[1] - 2*f[0] + 0)
            # Right boundary: no-flux (Neumann)
            f_new[-1] = f_new[-2]
            f = f_new
            f_all[n] = f
        return f_all

    def simulate_leapfrog(self):
        f_all = np.zeros((self.timesteps, self.N))
        f_all[0] = self.f0.copy()
        # First step: use upwind for n=1
        f_prev = self.f0.copy()
        f_curr = f_prev.copy()
        f_curr[1:] = f_prev[1:] - self.v * self.dt / self.da * (f_prev[1:] - f_prev[:-1])
        f_curr[0] = f_prev[0] - self.v * self.dt / self.da * (f_prev[0] - 0)
        f_curr[-1] = f_curr[-2]
        f_all[1] = f_curr
        # Leapfrog
        for n in range(2, self.timesteps):
            f_new = np.zeros_like(f_curr)
            f_new[1:-1] = f_prev[1:-1] - self.v * self.dt / self.da * (f_curr[2:] - f_curr[:-2])
            f_new[0] = f_prev[0] - self.v * self.dt / self.da * (f_curr[1] - 0)
            f_new[-1] = f_new[-2]
            f_all[n] = f_new
            f_prev, f_curr = f_curr, f_new
        return f_all

    def get_data(self):
        return self.f0, self.a, np.arange(self.timesteps) * self.dt 

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

    def get_data(self):
        return self.f0, self.a, np.arange(self.timesteps) * self.dt, self.G 