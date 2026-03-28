import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

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

def save_stitched_path_plot(name, gt_positions, output_dir, background_positions=None):
    if gt_positions.shape[0] >= 2:
        diffs = np.linalg.norm(gt_positions[1:] - gt_positions[:-1], axis=1)
        gap_indices = np.where(diffs > 5.0)[0] + 1
        if len(gap_indices) > 0:
            gt_positions = np.insert(gt_positions, gap_indices, np.nan, axis=0)
            
    fig = plt.figure(figsize=(20, 6), dpi=100)
    axes = [fig.add_subplot(1, 3, index + 1) for index in range(3)]
    projections = (
        (0, 2, 'x (m)', 'z (m)'),
        (0, 1, 'x (m)', 'y (m)'),
        (1, 2, 'y (m)', 'z (m)'),
    )
    
    if background_positions is not None:
        all_x = np.concatenate([background_positions[:, 0], gt_positions[:, 0]])
        all_y = np.concatenate([background_positions[:, 1], gt_positions[:, 1]])
        all_z = np.concatenate([background_positions[:, 2], gt_positions[:, 2]])
    else:
        all_x, all_y, all_z = gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2]

    for axis, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        if background_positions is not None:
            axis.plot(
                background_positions[:, x_idx],
                background_positions[:, y_idx],
                color='#888888',
                linewidth=1.0,
                label='TXT aligned to full_h5',
            )
        axis.plot(
            gt_positions[:, x_idx],
            gt_positions[:, y_idx],
            'r-',
            label='GT',
        )
        axis.plot(
            [gt_positions[0, x_idx]],
            [gt_positions[0, y_idx]],
            'ko',
            label='Start',
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(loc='upper right')
        
        arr_x = all_x if x_idx == 0 else (all_y if x_idx == 1 else all_z)
        arr_y = all_x if y_idx == 0 else (all_y if y_idx == 1 else all_z)
            
        _set_equal_axis_2d(axis, arr_x, arr_y)

    png_path = os.path.join(output_dir, f'{name}_full_route_path.png')
    pdf_path = os.path.join(output_dir, f'{name}_full_route_path.pdf')
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)

base_dir = "/home/ljc/Downloads/h5filewithturn/2-h5data"
out_dir = "/home/ljc/Projects/TransLO/experiment/h5filewithturn_viz"
os.makedirs(out_dir, exist_ok=True)

for seq in os.listdir(base_dir):
    seq_dir = os.path.join(base_dir, seq)
    if not os.path.isdir(seq_dir): continue
    
    base_map_path = os.path.join(seq_dir, f"{seq}xyz.txt")
    if not os.path.exists(base_map_path):
        print(f"Base map not found for {seq}, tried {base_map_path}")
        continue
        
    bg_positions = np.loadtxt(base_map_path)
    
    seq_out_dir = os.path.join(out_dir, seq)
    os.makedirs(seq_out_dir, exist_ok=True)
    
    for f in os.listdir(seq_dir):
        if f.endswith(".txt") and f != f"{seq}xyz.txt":
            gt_path = os.path.join(seq_dir, f)
            gt_positions = np.loadtxt(gt_path)
            
            name = f.replace(".txt", "")
            save_stitched_path_plot(name, gt_positions, seq_out_dir, background_positions=bg_positions)
            print(f"Saved path plot for {name} to {seq_out_dir}")
