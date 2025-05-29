import numpy as np
import pickle
import os
from Utils.Case4EQN import Case4EQN

# Parameters (ensure consistency with Case4EQN class)
a_max = 1.0
N = 200
dt = 0.002
t_max = 4.0

# Create Case4EQN instance
sim = Case4EQN(a_max=a_max, N=N, t_max=t_max, dt=dt)

# Run the LED simulation
f_all, t_vec = sim.simulate_led()

# Get plotting data (spatial variable 'a')
a_vec, _ = sim.get_data()

# Save results
case4_data = {
    "f_all": f_all,
    "t_vec": t_vec,
    "a_vec": a_vec,
}

os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/case4eqn_led_sim_data.pickle", "wb") as file:
    pickle.dump(case4_data, file, pickle.HIGHEST_PROTOCOL)

print("Case4EQN LED simulation data generated and saved.") 