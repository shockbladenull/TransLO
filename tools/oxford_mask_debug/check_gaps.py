import numpy as np
lo = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_LOxyz.txt")
diffs = np.linalg.norm(lo[1:] - lo[:-1], axis=1)
print(f"Max diff: {np.max(diffs):.2f}, mean diff: {np.mean(diffs):.2f}")
print(f"Number of gaps > 5m: {np.sum(diffs > 5.0)}")
