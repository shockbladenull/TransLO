import os
import numpy as np
import matplotlib.pyplot as plt

def get_segments(positions, threshold=5.0):
    if len(positions) == 0:
        return []
    diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    gap_indices = np.where(diffs > threshold)[0] + 1
    
    segments = []
    start = 0
    for idx in gap_indices:
        segments.append(positions[start:idx])
        start = idx
    segments.append(positions[start:])
    return segments

def _set_equal_axis_2d(ax, x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        return
    x_min, x_max = np.nanmin(x_values), np.nanmax(x_values)
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    x_mean = (x_min + x_max) / 2.0
    y_mean = (y_min + y_max) / 2.0
    plot_radius = max(x_max - x_min, y_max - y_min, 1e-6) / 2.0
    ax.set_xlim([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim([y_mean - plot_radius, y_mean + plot_radius])

def save_stitched_path_plot(name, lo_positions, scr_positions, output_dir, background_positions=None, draw_starts=True):
    fig = plt.figure(figsize=(20, 6), dpi=100)
    axes = [fig.add_subplot(1, 3, index + 1) for index in range(3)]
    projections = (
        (0, 2, 'x (m)', 'z (m)'),
        (0, 1, 'x (m)', 'y (m)'),
        (1, 2, 'y (m)', 'z (m)'),
    )
    
    lo_segments = get_segments(lo_positions)
    scr_segments = get_segments(scr_positions)
    
    if background_positions is not None:
        all_x = background_positions[:, 0]
        all_y = background_positions[:, 1]
        all_z = background_positions[:, 2]
    else:
        all_x = np.concatenate([lo_positions[:, 0], scr_positions[:, 0]])
        all_y = np.concatenate([lo_positions[:, 1], scr_positions[:, 1]])
        all_z = np.concatenate([lo_positions[:, 2], scr_positions[:, 2]])

    for axis, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        if background_positions is not None:
            axis.plot(
                background_positions[:, x_idx],
                background_positions[:, y_idx],
                color='#888888',
                linewidth=1.0,
                label='TXT aligned to full_h5'
            )
            
        lo_labeled = False
        for seg in lo_segments:
            if len(seg) == 0: continue
            axis.plot(seg[:, x_idx], seg[:, y_idx], 'r-', label='LO' if not lo_labeled else None)
            if draw_starts:
                axis.plot([seg[0, x_idx]], [seg[0, y_idx]], 'ko')
            lo_labeled = True
            
        scr_labeled = False
        for seg in scr_segments:
            if len(seg) == 0: continue
            axis.plot(seg[:, x_idx], seg[:, y_idx], 'g-', label='SCR' if not scr_labeled else None)
            if draw_starts:
                axis.plot([seg[0, x_idx]], [seg[0, y_idx]], 'ko')
            scr_labeled = True

        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axis.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        arr_x = all_x if x_idx == 0 else (all_y if x_idx == 1 else all_z)
        arr_y = all_x if y_idx == 0 else (all_y if y_idx == 1 else all_z)
            
        _set_equal_axis_2d(axis, arr_x, arr_y)

    png_path = os.path.join(output_dir, f'{name}_combo_full_route_path.png')
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

base_dir = "/home/ljc/Downloads/h5filewithturn/2-h5data"
out_dir = "/home/ljc/Projects/TransLO/experiment/h5filewithturn_combo_viz"
os.makedirs(out_dir, exist_ok=True)

for seq in os.listdir(base_dir):
    seq_dir = os.path.join(base_dir, seq)
    if not os.path.isdir(seq_dir): continue
    
    base_map_path = os.path.join(seq_dir, f"{seq}xyz.txt")
    if not os.path.exists(base_map_path): continue
    bg_positions = np.loadtxt(base_map_path)
    
    seq_out_dir = os.path.join(out_dir, seq)
    os.makedirs(seq_out_dir, exist_ok=True)
    
    lo_std = os.path.join(seq_dir, f"{seq}_LOxyz.txt")
    scr_std = os.path.join(seq_dir, f"{seq}_SCRxyz.txt")
    if os.path.exists(lo_std) and os.path.exists(scr_std):
        save_stitched_path_plot(f"{seq}_std", np.loadtxt(lo_std), np.loadtxt(scr_std), seq_out_dir, background_positions=bg_positions, draw_starts=True)
        print(f"Saved std combo for {seq}")

    lo_turn = os.path.join(seq_dir, f"{seq}_LOxyz_turning.txt")
    scr_turn = os.path.join(seq_dir, f"{seq}_SCRxyz_turning.txt")
    if os.path.exists(lo_turn) and os.path.exists(scr_turn):
        save_stitched_path_plot(f"{seq}_turning", np.loadtxt(lo_turn), np.loadtxt(scr_turn), seq_out_dir, background_positions=bg_positions, draw_starts=False)
        print(f"Saved turning combo for {seq}")
