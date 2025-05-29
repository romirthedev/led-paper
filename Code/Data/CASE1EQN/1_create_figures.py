import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import os

matplotlib.rcParams['text.usetex'] = False
plt.rcParams["text.usetex"] = False
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
font = {'weight':'normal', 'size':16}
plt.rc('font', **font)

with open("./Simulation_Data/case1eqn_fhnstyle.pickle", "rb") as file:
    data = pickle.load(file)
    rho_act_all = np.array(data["rho_act_all"])
    x = np.array(data["x"])
    t_vec_all = np.array(data["t_vec_all"])
    dt = data["dt"]
    del data

os.makedirs("./Figures", exist_ok=True)

# Plot initial condition
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, rho_act_all[0,0,:], label="Initial Condition", linewidth=2)
ax.set_ylabel(r"$f(a, t=0)$")
ax.set_xlabel(r"$a$")
ax.set_xlim([np.min(x), np.max(x)])
ax.set_title("Initial Condition for Equation 56")
ax.legend()
plt.tight_layout()
plt.savefig("./Figures/Plot_initial_condition.pdf")
plt.close()

# Plot final profile at t_final
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, rho_act_all[0,-1,:], label="Final Profile $f(a, t_{final})$", linewidth=2, color='black')
ax.set_ylabel(r"$f(a, t_{final})$")
ax.set_xlabel(r"$a$")
ax.set_xlim([np.min(x), np.max(x)])
ax.set_title("Final Profile for Equation 56 at $t_{final}$")
ax.legend()
plt.tight_layout()
plt.savefig("./Figures/Plot_final_profile.pdf")
plt.close()

print("CASE1EQN FHN-style 2D figures created in ./Figures/") 