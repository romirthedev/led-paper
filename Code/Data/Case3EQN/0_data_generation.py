import numpy as np
import pickle
import os
from Utils.Case3EQN import Case3EQN

# Parameters from Case3EQN class (ensure consistency or load dynamically)
a_max = 100.0
N = 400
dt = 0.01
t_max = 10.0

# Create Case3EQN instance
sim = Case3EQN(a_max=a_max, N=N, t_max=t_max, dt=dt)

# Run the LED simulation
f_all, t_vec = sim.simulate_led()

# Get plotting data (spatial variable 'a')
a_vec, _ = sim.get_data()

# Save results
case3_data = {
    "f_all": f_all,
    "t_vec": t_vec,
    "a_vec": a_vec,
}

os.makedirs("./Simulation_Data/", exist_ok=True)
with open("./Simulation_Data/case3eqn_led_sim_data.pickle", "wb") as file:
    pickle.dump(case3_data, file, pickle.HIGHEST_PROTOCOL)

print("Case3EQN LED simulation data generated and saved.") 