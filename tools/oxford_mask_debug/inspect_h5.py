import os
import h5py
import numpy as np

def inspect_folder(folder):
    files = os.listdir(folder)
    print(f"Contents of {folder}:", files)
    
    for f in files:
        path = os.path.join(folder, f)
        if f.endswith(".txt"):
            data = np.loadtxt(path)
            print(f"TXT {f}: shape={data.shape}")
        elif f.endswith(".h5"):
            with h5py.File(path, "r") as h5f:
                keys = list(h5f.keys())
                print(f"H5 {f}: keys={keys}, poses_shape={h5f['poses'].shape if 'poses' in h5f else 'N/A'}")

inspect_folder("/home/ljc/Downloads/h5filewithturn/2-h5data/0226")
