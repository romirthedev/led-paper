import numpy as np
import pickle
import os
from Utils.CASE1EQN import CASE1EQN, CASE2EQN

# Parameters from equation 56
a_max = 1.0
N = 200
da = a_max / N
v = 0.1
dt = da / v  # CFL=1
t_max = 4.0

# Only one initial condition as in equation 56
init_conditions = [lambda a: 100 * np.exp(-a / 0.01)]

f_all = []
t_vec_all = []
a_vec = None
for f0_func in init_conditions:
    sim = CASE1EQN(a_max=a_max, N=N, t_max=t_max, dt=dt, v=v)
    sim.f0 = f0_func(sim.a)
    f = sim.simulate_upwind()  # Only upwind for this case
    f_all.append(f)
    t_vec_all.append(np.arange(sim.timesteps) * dt)
    a_vec = sim.a

# Save results in FHN-style dict
# (use 'rho_act_all' for the main variable, 't_vec_all', and 'x' for a)
data = {
    "rho_act_all": f_all,  # shape: (num_ic, timesteps, N)
    "t_vec_all": t_vec_all,  # shape: (num_ic, timesteps)
    "x": a_vec,  # spatial grid
    "dt": dt,
    "N": N,
    "L": a_max,
    "dx": da,
    "v": v,
    "init_condition": "100*exp(-a/0.01)",
}
os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/case1eqn_fhnstyle.pickle", "wb") as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

# --- Case 2: Equation 57 ---
a_max2 = 1.0
N2 = 200
da2 = a_max2 / N2
dt2 = 0.002  # Use a reasonable dt for stability
t_max2 = 4.0

sim2 = CASE2EQN(a_max=a_max2, N=N2, t_max=t_max2, dt=dt2)
f2 = sim2.simulate_upwind()
f0_2, a2, t2, G2 = sim2.get_data()

# Save results in FHN-style dict
case2_data = {
    "rho_act_all": [f2],
    "t_vec_all": [np.arange(sim2.timesteps) * dt2],
    "x": a2,
    "dt": dt2,
    "N": N2,
    "L": a_max2,
    "dx": da2,
    "G": G2,
    "init_condition": "50*exp(-(a-0.2)**2/0.0005)",
}
os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/case2eqn_fhnstyle.pickle", "wb") as file:
    pickle.dump(case2_data, file, pickle.HIGHEST_PROTOCOL) 