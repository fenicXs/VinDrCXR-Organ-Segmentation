"""
Monitor training progress by checking logs and GPU usage
"""

import os
import time
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'memory_used': float(info[1]),
                'memory_total': float(info[2]),
                'gpu_util': float(info[3])
            }
    except:
        return None
    return None

def check_checkpoint_dir():
    """Check for saved checkpoints"""
    checkpoint_dir = "checkpoints/swin_seg_VinDr-CXR"
    if os.path.exists(checkpoint_dir):
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        return files
    return []

def print_status():
    """Print current training status"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print("CheXmask Swin-UNet Training Monitor")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    gpu_info = get_gpu_info()
    if gpu_info:
        print("GPU Status:")
        print(f"  Device: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['memory_used']:.0f} / {gpu_info['memory_total']:.0f} MB ({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
        print(f"  Utilization: {gpu_info['gpu_util']:.0f}%")
        print()
    
    checkpoints = check_checkpoint_dir()
    if checkpoints:
        print(f"Checkpoints Found: {len(checkpoints)}")
        for ckpt in sorted(checkpoints):
            print(f"  - {ckpt}")
        print()
    else:
        print("No checkpoints yet...")
        print()
    
    output_dir = "outputs/swin_seg_VinDr-CXR"
    if os.path.exists(output_dir):
        vis_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if vis_files:
            print(f"Visualizations: {len(vis_files)}")
            for vis in sorted(vis_files)[-3:]:  # Show last 3
                print(f"  - {vis}")
            print()
    
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)

if __name__ == "__main__":
    print("Starting training monitor...")
    print("Monitoring every 30 seconds...")
    print()
    
    try:
        while True:
            print_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("\nFinal Status:")
        print_status()
