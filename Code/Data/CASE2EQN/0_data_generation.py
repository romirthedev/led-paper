import numpy as np
import pickle
import os
from Utils.CASE2EQN import CASE2EQN

# Parameters for equation 57
a_max = 1.0
N = 200
da = a_max / N
dt = 0.002  # Use a reasonable dt for stability
t_max = 4.0

sim = CASE2EQN(a_max=a_max, N=N, t_max=t_max, dt=dt)
# Use the LED simulation method
f_all, t_vec = sim.simulate_led()

f0, a_vec, _, G = sim.get_data() # get_data also returns t_vec, but we use the one from simulate_led

# Save results in FHN-style dict
# The output of simulate_led is f_all (shape timesteps x N) and t_vec (shape timesteps)
case2_data = {
    "rho_act_all": [f_all],
    "t_vec_all": [t_vec],
    "x": a_vec,
    "dt": dt,
    "N": N,
    "L": a_max,
    "dx": da,
    "G": G,
    "init_condition": "50*exp(-(a-0.2)**2/0.0005)",
}
os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/case2eqn_fhnstyle.pickle", "wb") as file:
    pickle.dump(case2_data, file, pickle.HIGHEST_PROTOCOL) 