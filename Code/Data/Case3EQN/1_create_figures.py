import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Load simulation data
data_path = "./Simulation_Data/case3eqn_led_sim_data.pickle"
with open(data_path, "rb") as file:
    data = pickle.load(file)

f_all = data["f_all"]
t_vec = data["t_vec"]
a_vec = data["a_vec"]

# Extract final profile and time
final_profile = f_all[-1, :]
final_time = t_vec[-1]

# Plot final profile
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(a_vec, final_profile, label=f'Final Profile at t = {final_time:.2f}', linewidth=2)
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$f(a, t_{final})$')
ax.set_title('Case 3 Equation Final Profile (LED Simulation)')
ax.legend()
ax.grid(True)

# Save the figure
os.makedirs("./Figures/", exist_ok=True)
plt.tight_layout()
plt.savefig("./Figures/Plot_final_profile_case3_led.pdf")
plt.close()

print("Case3EQN final profile figure (LED) created in ./Figures/Plot_final_profile_case3_led.pdf") 