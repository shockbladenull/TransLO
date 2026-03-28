import numpy as np

def compare_splits():
    scr = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_SCRxyz.txt")
    scr_t = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_SCRxyz_turning.txt")
    lo = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_LOxyz.txt")
    lo_t = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226_LOxyz_turning.txt")
    full = np.loadtxt("/home/ljc/Downloads/h5filewithturn/2-h5data/0226/0226xyz.txt")
    
    print(f"Full: {len(full)}")
    print(f"SCR + LO: {len(scr)} + {len(lo)} = {len(scr) + len(lo)}")
    print(f"SCR_turning + LO_turning: {len(scr_t)} + {len(lo_t)} = {len(scr_t) + len(lo_t)}")
    
compare_splits()
