import numpy as np
import pickle
import os
import h5py

# Load simulation data
with open("./Simulation_Data/kse_sim.pickle", "rb") as file:
    simdata = pickle.load(file)
    u = np.array(simdata["u"])
    L = np.array(simdata["L"])
    N = np.array(simdata["N"])
    dt = simdata["dt"]
    del simdata

# Downsample and reshape for consistency
trajectory = u[:, np.newaxis]

# Save scaler values and dt
os.makedirs("./Data", exist_ok=True)
data_max = np.max(trajectory)
data_min = np.min(trajectory)
np.savetxt("./Data/data_max.txt", [data_max])
np.savetxt("./Data/data_min.txt", [data_min])
np.savetxt("./Data/dt.txt", [dt])

# Split data into train/val/test (as evenly as possible)
num_timesteps = trajectory.shape[0]
num_splits = 3
split_sizes = [num_timesteps // num_splits + (1 if x < num_timesteps % num_splits else 0) for x in range(num_splits)]
split_indices = np.cumsum([0] + split_sizes)
split_names = ["train", "val", "test"]

sequence_length = min(50, num_timesteps // 4)  # Use a reasonable sequence length
batch_size = 4  # Try to create up to 4 batches per split

for i, split_name in enumerate(split_names):
    split_data = trajectory[split_indices[i]:split_indices[i+1]]
    n_timesteps = split_data.shape[0]
    if n_timesteps <= sequence_length:
        print(f"Skipping {split_name}: not enough time steps ({n_timesteps}) for sequence length {sequence_length}.")
        continue
    n_possible_batches = n_timesteps - sequence_length
    n_batches = min(batch_size, n_possible_batches)
    print(f"{split_name}: {n_timesteps} time steps, {n_possible_batches} possible batches of length {sequence_length}.")
    data_dir = f"./Data/{split_name}"
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    idxs = np.arange(n_possible_batches)
    for seq_num in range(n_batches):
        idx = np.random.choice(idxs, replace=False)
        sequence = split_data[idx:idx + sequence_length]
        print(f'batch_{seq_num:06d}', np.shape(sequence))
        gg = hf.create_group(f'batch_{seq_num:06d}')
        gg.create_dataset('data', data=sequence)
        idxs = idxs[idxs != idx]  # Remove used idx
    hf.close()
    # Save raw data for SINDy
    data_dir_raw = f"./Data/{split_name}_raw"
    os.makedirs(data_dir_raw, exist_ok=True)
    hf = h5py.File(data_dir_raw + '/data.h5', 'w')
    data_group = split_data
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(0))
    gg.create_dataset('data', data=data_group)
    hf.close()

print("Done! Data splits and sequence creation complete.") 