import numpy as np
import matplotlib.pyplot as plt

lo_std = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_LOxyz.txt")
lo_turn = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_LOxyz_turning.txt")
scr_std = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_SCRxyz.txt")
scr_turn = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_SCRxyz_turning.txt")
base_map = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226xyz.txt")

lo_s = set(tuple(np.round(p, 4)) for p in lo_std)
lo_t = set(tuple(np.round(p, 4)) for p in lo_turn)
scr_s = set(tuple(np.round(p, 4)) for p in scr_std)
scr_t = set(tuple(np.round(p, 4)) for p in scr_turn)

lo_only_std = np.array(list(lo_s - lo_t))
lo_only_turn = np.array(list(lo_t - lo_s))

print(f"Points in LO_std but not in LO_turn: {len(lo_only_std)}")
print(f"Points in LO_turn but not in LO_std: {len(lo_only_turn)}")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(base_map[:, 0], base_map[:, 1], color='#dddddd', label='Base')

if len(lo_only_std) > 0:
    ax.scatter(lo_only_std[:, 0], lo_only_std[:, 1], c='blue', s=10, label='Removed from LO in turning (Now in SCR)')
if len(lo_only_turn) > 0:
    ax.scatter(lo_only_turn[:, 0], lo_only_turn[:, 1], c='red', s=10, label='Added to LO in turning (Was in SCR)')

ax.legend()
fig.savefig('/home/ljc/Projects/TransLO/experiment/diff_plot.png')
print("Diff plot saved.")
