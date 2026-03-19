import os
import argparse
import csv

def export_scalars_to_csv(event_file_path, output_dir=None):
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    if not os.path.exists(event_file_path):
        print(f"Error: File '{event_file_path}' does not exist.")
        return

    if output_dir is None:
        output_dir = os.path.dirname(event_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading events from {event_file_path}...")
    loader = LegacyEventFileLoader(event_file_path)
    
    data = {}
    
    print("Parsing events, this might take a moment...")
    
    for event in loader.Load():
        if event.summary is not None:
            for value in event.summary.value:
                # PyTorch classic scalars
                val = None
                if value.HasField('simple_value'):
                    val = value.simple_value
                elif value.HasField('tensor') and value.metadata.plugin_data.plugin_name == 'scalars':
                    # PyTorch 1.x/2.x tensor scalars
                    try:
                        from tensorboard.util import tensor_util
                        val = tensor_util.make_ndarray(value.tensor).item()
                    except BaseException:
                        pass
                
                if val is not None:
                    tag = value.tag
                    if tag not in data:
                        data[tag] = []
                    data[tag].append((event.wall_time, event.step, val))
    
    if not data:
        print("No scalars found in the event file.")
        return

    print(f"Found {len(data)} scalar tags.")

    for tag, points in data.items():
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f"{safe_tag}.csv")
        
        with open(output_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wall_time', 'step', 'value'])
            for pt in points:
                writer.writerow(pt)
                
        print(f"Saved tag '{tag}' to '{output_file}' with {len(points)} values.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TensorBoard scalars to CSV')
    parser.add_argument('event_file', type=str, help='Path to the tfevents file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for CSV files (defaults to the event file directory)')
    
    args = parser.parse_args()
    export_scalars_to_csv(args.event_file, args.output_dir)
