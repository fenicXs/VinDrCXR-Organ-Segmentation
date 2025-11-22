"""
Create a detailed visualization viewer for the training results
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

output_dir = "outputs/swin_seg_VinDr-CXR"
curves_path = os.path.join(output_dir, "training_curves.png")
predictions_path = os.path.join(output_dir, "epoch_1_predictions.png")

print("=" * 80)
print("VISUALIZING TRAINING RESULTS")
print("=" * 80)
print()

fig = plt.figure(figsize=(20, 10))

if os.path.exists(curves_path):
    print(f"✓ Found training curves: {curves_path}")
    ax1 = plt.subplot(1, 2, 1)
    img1 = mpimg.imread(curves_path)
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('Training Curves (Loss & Metrics)', fontsize=14, fontweight='bold')
else:
    print(f"✗ Training curves not found: {curves_path}")

if os.path.exists(predictions_path):
    print(f"✓ Found predictions: {predictions_path}")
    ax2 = plt.subplot(1, 2, 2)
    img2 = mpimg.imread(predictions_path)
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('Epoch 1 Predictions', fontsize=14, fontweight='bold')
else:
    print(f"✗ Predictions not found: {predictions_path}")

plt.tight_layout()
output_combined = os.path.join(output_dir, "combined_results_view.png")
plt.savefig(output_combined, dpi=150, bbox_inches='tight')
print()
print(f"✓ Saved combined visualization to: {output_combined}")
print()

print("=" * 80)
print("INDIVIDUAL VISUALIZATIONS")
print("=" * 80)
print()

if os.path.exists(curves_path):
    fig2 = plt.figure(figsize=(12, 8))
    img = mpimg.imread(curves_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training Curves - Loss, Dice, and IoU over Epochs', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    curves_large = os.path.join(output_dir, "training_curves_large.png")
    plt.savefig(curves_large, dpi=200, bbox_inches='tight')
    print(f"✓ Saved large training curves to: {curves_large}")
    plt.close()

if os.path.exists(predictions_path):
    fig3 = plt.figure(figsize=(16, 10))
    img = mpimg.imread(predictions_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Sample Predictions from Epoch 1\n(Input | Ground Truth | Prediction)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    pred_large = os.path.join(output_dir, "predictions_large.png")
    plt.savefig(pred_large, dpi=200, bbox_inches='tight')
    print(f"✓ Saved large predictions to: {pred_large}")
    plt.close()

print()
print("=" * 80)
print("All visualizations created successfully!")
print("=" * 80)
