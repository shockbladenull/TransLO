import numpy as np

def count_segments(positions, threshold=5.0):
    if len(positions) == 0:
        return 0
    diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    gaps = np.sum(diffs > threshold)
    return gaps + 1

def analyze():
    base_dir = "/home/ljc/Downloads/h5filewithturn/2-h5data/0226"
    
    lo_std = np.loadtxt(f"{base_dir}/0226_LOxyz.txt")
    scr_std = np.loadtxt(f"{base_dir}/0226_SCRxyz.txt")
    
    lo_turn = np.loadtxt(f"{base_dir}/0226_LOxyz_turning.txt")
    scr_turn = np.loadtxt(f"{base_dir}/0226_SCRxyz_turning.txt")
    
    print("--- 0226 Std ---")
    print(f"LO std segments:  {count_segments(lo_std)}")
    print(f"SCR std segments: {count_segments(scr_std)}")
    print(f"Total std dots:   {count_segments(lo_std) + count_segments(scr_std)}\n")
    
    print("--- 0226 Turning ---")
    print(f"LO turn segments:  {count_segments(lo_turn)}")
    print(f"SCR turn segments: {count_segments(scr_turn)}")
    print(f"Total turn dots:   {count_segments(lo_turn) + count_segments(scr_turn)}\n")

analyze()
