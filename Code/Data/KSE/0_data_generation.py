import numpy as np
import pickle
import os
from Utils.KSE import KSE

# Define parameters
L = 22 / (2 * np.pi)
N = 64
dt = 0.0005  # Larger timestep
# Shorter simulation
# 120/0.0005 = 240,000 timesteps
# This is much more manageable

tend = 120

# Initialize and simulate
kse = KSE(L=L, N=N, dt=dt, tend=tend, nu=0.01)
kse.simulate()
kse.fou2real()

# Get field and downsample in time
u = kse.uu[::100]  # Save every 100th timestep

# Save results
data = {
    "u": u,
    "L": L,
    "N": N,
    "dt": dt,
}

os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/kse_sim.pickle", "wb") as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL) 