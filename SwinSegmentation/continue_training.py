"""
Quick script to continue training from the best checkpoint
"""

import os
import sys

print("=" * 80)
print("CONTINUE TRAINING FROM CHECKPOINT")
print("=" * 80)
print()

checkpoint_path = "checkpoints/swin_seg_VinDr-CXR/best_model.pth"
if os.path.exists(checkpoint_path):
    print(f"✓ Found best model checkpoint: {checkpoint_path}")
    print(f"  - Trained for: 3 epochs")
    print(f"  - Best Val Dice: 0.1144 (11.44%)")
    print()
else:
    print(f"✗ Checkpoint not found: {checkpoint_path}")
    print("  Please train the model first with: python train.py")
    sys.exit(1)

print("Current Configuration:")
print("-" * 80)
print("To extend training, update config.py:")
print()
print("  NUM_EPOCHS = 100  # Change from 3 to 100")
print()
print("Then run:")
print("  python train.py --resume")
print()
print("Note: The training script will automatically resume from the best checkpoint")
print("      if it detects existing checkpoints for the same experiment.")
print()

epochs_remaining = 97  # 100 - 3
time_per_epoch = 37  # minutes
total_hours = (epochs_remaining * time_per_epoch) / 60

print("Estimated Training Time:")
print("-" * 80)
print(f"  Remaining epochs: {epochs_remaining}")
print(f"  Time per epoch:   ~{time_per_epoch} minutes")
print(f"  Total time:       ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")
print()

print("=" * 80)
print("💡 Recommendations:")
print("=" * 80)
print()
print("1. Update config.py to NUM_EPOCHS = 100")
print("2. Run training with: python train.py")
print("3. Monitor progress with: python check_progress.py")
print("4. Training will auto-resume from best checkpoint")
print("5. Best model will be saved automatically")
print()
print("=" * 80)
