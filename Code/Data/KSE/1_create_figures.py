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

with open("./Simulation_Data/kse_sim.pickle", "rb") as file:
    data = pickle.load(file)
    u = np.array(data["u"])  # (timesteps, N)
    L = data["L"]
    N = data["N"]
    dt = data["dt"]
    del data

x = np.linspace(0, L, N, endpoint=False)
t_vec = np.arange(u.shape[0]) * dt

os.makedirs("./Figures", exist_ok=True)

# Plot initial condition
plt.figure(figsize=(8,5))
plt.plot(x, u[0], label="Initial condition", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x, t=0)$")
plt.title("KSE Initial Condition")
plt.legend()
plt.tight_layout()
plt.savefig("./Figures/Plot_initial_condition.pdf")
plt.close()

# Surface and contour plots for the full simulation
subsample_time = max(1, u.shape[0] // 100)  # Show up to 100 time slices
X, Y = np.meshgrid(x, t_vec[::subsample_time])
Z = u[::subsample_time]

# 3D surface plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rasterized=True)
ax.set_xlabel(r"$x$", labelpad=20)
ax.set_ylabel(r"$t$", labelpad=20)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r"$u(x,t)$", rotation=0, labelpad=20)
fig.colorbar(surf, orientation="horizontal")
ax.invert_xaxis()
ax.view_init(elev=34., azim=-48.)
plt.savefig("./Figures/Plot_surface.pdf")
plt.close()

# Contour plot
fig = plt.figure()
ax = fig.gca()
mp = ax.contourf(X, Y, Z, 100, cmap=plt.get_cmap("seismic"), zorder=-9)
ax.set_ylabel(r"$t$")
ax.set_xlabel(r"$x$")
fig.colorbar(mp)
plt.gca().set_rasterization_zorder(-1)
plt.tight_layout()
plt.savefig("./Figures/Plot_contourf.pdf")
plt.close()

print("KSE figures created in ./Figures/") 