import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
data_path = "./Simulation_Data/case2eqn_fhnstyle.pickle"
with open(data_path, "rb") as file:
    data = pickle.load(file)

rho_act_all = data["rho_act_all"][0]
t_vec = data["t_vec_all"][0]
x = data["x"]

# Plot initial condition
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, rho_act_all[0,:], label="Initial Condition", linewidth=2)
ax.set_ylabel(r"$f(a, t=0)$")
ax.set_xlabel(r"$a$")
ax.set_xlim([np.min(x), np.max(x)])
ax.set_title("Initial Condition for Equation 57")
ax.legend()
plt.tight_layout()
os.makedirs("./Figures/", exist_ok=True)
plt.savefig("./Figures/Plot_initial_condition.pdf")
plt.close()

# Plot final profile at t_final
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, rho_act_all[-1,:], label="Final Profile $f(a, t_{final})$", linewidth=2, color='black')
ax.set_ylabel(r"$f(a, t_{final})$")
ax.set_xlabel(r"$a$")
ax.set_xlim([np.min(x), np.max(x)])
ax.set_title("Final Profile for Equation 57 at $t_{final}$ (LED simulation)")
ax.legend()
plt.tight_layout()
plt.savefig("./Figures/Plot_final_profile_led.pdf")
plt.close()

print("CASE2EQN final profile figure (LED) created in ./Figures/") 