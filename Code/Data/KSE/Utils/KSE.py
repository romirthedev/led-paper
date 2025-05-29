import numpy as np
from numpy.fft import fft, ifft, fftfreq
from tqdm import tqdm

class KSE:
    def __init__(self, L, N, dt, tend, nu=0.01):
        self.L = L
        self.N = N
        self.dt = dt
        self.tend = tend
        self.nu = nu
        self.x = np.linspace(0, L, N, endpoint=False)
        self.k = fftfreq(N, d=L/N) * 2 * np.pi
        self.u0 = np.cos(2 * np.pi * self.x / L)  # Example initial condition
        self.timesteps = int(tend / dt)
        self.uu = np.zeros((self.timesteps, N))

    def simulate(self):
        u = self.u0.copy()
        for i in tqdm(range(self.timesteps), desc='KSE Simulation'):
            u_hat = fft(u)
            du_dx = np.real(ifft(1j * self.k * u_hat))
            d2u_dx2 = np.real(ifft(-(self.k**2) * u_hat))
            du_dt = self.nu * d2u_dx2 - u * du_dx
            u = u + self.dt * du_dt
            self.uu[i] = u
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                print(f"Simulation stopped early at step {i} due to instability.")
                self.uu = self.uu[:i+1]
                break

    def fou2real(self):
        # Already in real space after simulate
        pass 