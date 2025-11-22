"""
Quick training progress check
"""
import os
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'memory_used': float(info[1]),
                'memory_total': float(info[2]),
                'gpu_util': float(info[3]),
                'temp': float(info[4])
            }
    except:
        pass
    return None

print("=" * 70)
print("TRAINING STATUS CHECK")
print("=" * 70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

gpu_info = get_gpu_info()
if gpu_info:
    print("🖥️  GPU STATUS:")
    print(f"   Device: {gpu_info['name']}")
    print(f"   Memory: {gpu_info['memory_used']:.0f} / {gpu_info['memory_total']:.0f} MB " +
          f"({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
    print(f"   Utilization: {gpu_info['gpu_util']:.0f}%")
    print(f"   Temperature: {gpu_info['temp']:.0f}°C")
    print()

checkpoint_dir = "checkpoints/swin_seg_VinDr-CXR"
if os.path.exists(checkpoint_dir):
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    if files:
        print(f"💾 CHECKPOINTS: {len(files)} saved")
        for f in files:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, f)) / (1024*1024)
            print(f"   ✓ {f} ({size_mb:.1f} MB)")
        print()
    else:
        print("💾 No checkpoints yet (training in progress...)\n")
else:
    print("💾 Checkpoint directory not created yet\n")

output_dir = "outputs/swin_seg_VinDr-CXR"
if os.path.exists(output_dir):
    vis_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    if vis_files:
        print(f"📊 VISUALIZATIONS: {len(vis_files)} created")
        for f in vis_files[-3:]:  # Show last 3
            print(f"   ✓ {f}")
        print()

print("=" * 70)
print("💡 Tips:")
print("   - Run 'python monitor_training.py' for live monitoring")
print("   - Check 'outputs/swin_seg_VinDr-CXR/' for prediction visualizations")
print("   - Checkpoints are saved after each epoch")
print("=" * 70)
