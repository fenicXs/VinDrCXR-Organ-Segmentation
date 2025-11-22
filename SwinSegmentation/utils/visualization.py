"""
Visualization utilities for segmentation
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from matplotlib.colors import ListedColormap


def visualize_segmentation(image, mask, pred_mask=None, alpha=0.4, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        image: (H, W) - grayscale image
        mask: (H, W) - ground truth mask
        pred_mask: (H, W) - predicted mask (optional)
        alpha: overlay transparency
        save_path: path to save the visualization
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if pred_mask is not None and torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    if image.max() > 1.0:
        image = image / 255.0
    
    colors = np.array([
        [0, 0, 0],      # Background - black
        [1, 0, 0],      # Right Lung - red
        [0, 1, 0],      # Left Lung - green
        [0, 0, 1],      # Heart - blue
    ])
    
    def mask_to_rgb(mask):
        """Convert class mask to RGB"""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3))
        for i in range(4):
            rgb[mask == i] = colors[i]
        return rgb
    
    mask_rgb = mask_to_rgb(mask)
    
    if pred_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        pred_mask_rgb = mask_to_rgb(pred_mask)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    image_rgb = np.stack([image, image, image], axis=-1)
    overlay_gt = image_rgb * (1 - alpha) + mask_rgb * alpha
    axes[2].imshow(overlay_gt)
    axes[2].set_title('GT Overlay')
    axes[2].axis('off')
    
    if pred_mask is not None:
        overlay_pred = image_rgb * (1 - alpha) + pred_mask_rgb * alpha
        axes[3].imshow(overlay_pred)
        axes[3].set_title('Prediction Overlay')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(images, masks, pred_masks=None, num_samples=4, save_path=None):
    """
    Visualize a batch of samples
    
    Args:
        images: (B, 1, H, W) - batch of images
        masks: (B, H, W) - batch of ground truth masks
        pred_masks: (B, H, W) - batch of predicted masks (optional)
        num_samples: number of samples to visualize
        save_path: path to save the visualization
    """
    num_samples = min(num_samples, images.shape[0])
    
    if pred_masks is not None:
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    else:
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = images[i, 0].cpu().numpy()
        mask = masks[i].cpu().numpy()
        
        if img.max() > 1.0:
            img = img / 255.0
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='jet', vmin=0, vmax=3)
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 1].axis('off')
        
        if pred_masks is not None:
            pred = pred_masks[i].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='jet', vmin=0, vmax=3)
            axes[i, 2].set_title(f'Sample {i+1}: Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: dictionary with training history
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_dice'], label='Train')
    axes[0, 1].plot(history['val_dice'], label='Validation')
    axes[0, 1].set_title('Mean Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['train_iou'], label='Train')
    axes[1, 0].plot(history['val_iou'], label='Validation')
    axes[1, 0].set_title('Mean IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(image, gt_mask, pred_mask, class_names=None, save_path=None):
    """
    Create a detailed comparison grid showing per-class results
    
    Args:
        image: (H, W) - grayscale image
        gt_mask: (H, W) - ground truth mask
        pred_mask: (H, W) - predicted mask
        class_names: list of class names
        save_path: path to save the visualization
    """
    if class_names is None:
        class_names = ['Background', 'Right Lung', 'Left Lung', 'Heart']
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 4, figsize=(16, 4 * num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for c in range(num_classes):
        gt_binary = (gt_mask == c).astype(np.uint8)
        pred_binary = (pred_mask == c).astype(np.uint8)
        
        tp = np.logical_and(gt_binary, pred_binary).sum()
        fp = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
        fn = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        
        overlay_gt = np.stack([image, image, image], axis=-1).copy()
        overlay_gt[gt_binary == 1] = [1, 0, 0]  # Red for GT
        axes[c, 0].imshow(overlay_gt)
        axes[c, 0].set_title(f'{class_names[c]} - GT')
        axes[c, 0].axis('off')
        
        overlay_pred = np.stack([image, image, image], axis=-1).copy()
        overlay_pred[pred_binary == 1] = [0, 1, 0]  # Green for Pred
        axes[c, 1].imshow(overlay_pred)
        axes[c, 1].set_title(f'{class_names[c]} - Pred')
        axes[c, 1].axis('off')
        
        error_map = np.zeros_like(overlay_gt)
        error_map[np.logical_and(gt_binary, pred_binary) == 1] = [1, 1, 1]  # TP - white
        error_map[np.logical_and(pred_binary, np.logical_not(gt_binary)) == 1] = [1, 0, 0]  # FP - red
        error_map[np.logical_and(np.logical_not(pred_binary), gt_binary) == 1] = [0, 0, 1]  # FN - blue
        axes[c, 2].imshow(error_map)
        axes[c, 2].set_title(f'{class_names[c]} - Errors')
        axes[c, 2].axis('off')
        
        axes[c, 3].text(0.1, 0.6, f'Dice: {dice:.4f}', fontsize=14)
        axes[c, 3].text(0.1, 0.4, f'IoU: {iou:.4f}', fontsize=14)
        axes[c, 3].text(0.1, 0.2, f'TP: {tp}  FP: {fp}  FN: {fn}', fontsize=10)
        axes[c, 3].set_xlim([0, 1])
        axes[c, 3].set_ylim([0, 1])
        axes[c, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_prediction_as_mask(pred_mask, save_path, height=None, width=None):
    """
    Save prediction mask as an image
    
    Args:
        pred_mask: (H, W) - predicted mask
        save_path: path to save the mask
        height: target height (optional)
        width: target width (optional)
    """
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    if height is not None and width is not None:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(save_path, pred_mask.astype(np.uint8))


def masks_to_rgb(masks):
    """
    Convert batch of class masks to RGB for visualization
    
    Args:
        masks: (B, H, W) - batch of masks
        
    Returns:
        rgb: (B, H, W, 3) - RGB visualization
    """
    colors = np.array([
        [0, 0, 0],      # Background - black
        [255, 0, 0],    # Right Lung - red
        [0, 255, 0],    # Left Lung - green
        [0, 0, 255],    # Heart - blue
    ], dtype=np.uint8)
    
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    B, H, W = masks.shape
    rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
    
    for i in range(4):
        rgb[masks == i] = colors[i]
    
    return rgb
