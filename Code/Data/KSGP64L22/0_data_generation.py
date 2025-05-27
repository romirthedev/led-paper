import numpy as np
from numpy import pi
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d
from Utils import KS
import gc
import h5py
import os

#------------------------------------------------------------------------------
# define data and initialize simulation
L    = 22/(2*pi)
N    = 64
dt   = 0.00025

# Process in smaller chunks
chunk_size = 1000000  # Process 1 million steps at a time
ttransient = 1000
tend = 12000 + ttransient
total_steps = int(tend/dt)

# Initialize arrays to store results
u_chunks = []

# Process in chunks
for chunk_start in range(0, total_steps, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total_steps)
    print(f"Processing chunk {chunk_start//chunk_size + 1} of {total_steps//chunk_size + 1}")
    
    # Initialize simulation for this chunk
    dns = KS.KS(L=L, N=N, dt=dt, tend=(chunk_end-chunk_start)*dt)
    
    # Simulate this chunk
    dns.simulate()
    dns.fou2real()
    
    # Get field for this chunk
    u_chunk = dns.uu
    
    # Remove initial transients if this is the first chunk
    if chunk_start == 0:
        ntransientsteps = int(ttransient/dt)
        u_chunk = u_chunk[ntransientsteps:]
    
    # Store chunk
    u_chunks.append(u_chunk)
    
    # Clear memory
    del dns
    gc.collect()
    
    # Save intermediate results
    if chunk_start % (chunk_size * 5) == 0:  # Save every 5 chunks
        intermediate_data = {
            "u": np.concatenate(u_chunks),
            "L": L,
            "N": N,
            "dt": dt,
        }
        with open("./Simulation_Data/ks_sim_intermediate.pickle", "wb") as file:
            pickle.dump(intermediate_data, file, pickle.HIGHEST_PROTOCOL)
        del intermediate_data
        gc.collect()

# Combine all chunks
u = np.concatenate(u_chunks)
del u_chunks
gc.collect()

# Save final results
data = {
    "u": u,
    "L": L,
    "N": N,
    "dt": dt,
}

os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/ks_sim.pickle", "wb") as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

# Clear final memory
del data
del u
gc.collect()



