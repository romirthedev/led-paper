# Dynamically determine splits and sequence lengths based on available data
num_timesteps = trajectory.shape[0]

# Use 3 splits, each with enough data for at least one sequence
min_seq_len = 10
split = max(num_timesteps // 3, min_seq_len + 1)

train_end = split
val_end = 2 * split

data_train = trajectory[:train_end]
data_val = trajectory[train_end:val_end]
data_test = trajectory[val_end:]

splits = [data_train, data_val, data_test]
split_names = ["train", "val", "test"]

for split_data, split_name in zip(splits, split_names):
    n_timesteps = split_data.shape[0]
    if n_timesteps <= min_seq_len:
        print(f"Skipping {split_name}: not enough time steps ({n_timesteps}) for sequence length {min_seq_len}.")
        continue
    n_possible_ics = n_timesteps - min_seq_len
    n_ics = min(2, n_possible_ics)  # At most 2 initial conditions
    print(f"{split_name}: {n_timesteps} time steps, {n_possible_ics} possible initial conditions for sequence length {min_seq_len}.")
    data_dir = f"./Data/{split_name}"
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    idxs = np.arange(n_possible_ics)
    for seq_num in range(n_ics):
        idx = np.random.choice(idxs)
        sequence = split_data[idx:idx + min_seq_len]
        print(f'batch_{seq_num:06d}', np.shape(sequence))
        gg = hf.create_group(f'batch_{seq_num:06d}')
        gg.create_dataset('data', data=sequence)
    hf.close()
    # Creating raw data sequence (needed for training SINDy)
    data_dir_raw = f"./Data/{split_name}_raw"
    os.makedirs(data_dir_raw, exist_ok=True)
    hf = h5py.File(data_dir_raw + '/data.h5', 'w')
    data_group = split_data
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(0))
    gg.create_dataset('data', data=data_group)
    hf.close()

print("Done! Data splits and sequence creation complete.") 