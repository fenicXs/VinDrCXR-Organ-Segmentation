"""
Loss functions and metrics for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    def __init__(self, smooth=1e-5, class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, H, W) - ground truth labels
        """
        pred = F.softmax(pred, dim=1)
        
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=pred.device)
            dice = dice * weights.unsqueeze(0)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = DiceLoss(class_weights=class_weights)
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(config):
    """Get loss function based on config"""
    loss_type = config.LOSS_TYPE
    class_weights = config.CLASS_WEIGHTS
    
    if loss_type == 'ce':
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'dice':
        return DiceLoss(class_weights=class_weights)
    
    elif loss_type == 'combined':
        return CombinedLoss(
            ce_weight=config.CE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            class_weights=class_weights
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class SegmentationMetrics:
    """
    Calculate segmentation metrics
    """
    def __init__(self, num_classes, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.tp = torch.zeros(self.num_classes, device=self.device)
        self.fp = torch.zeros(self.num_classes, device=self.device)
        self.fn = torch.zeros(self.num_classes, device=self.device)
        self.tn = torch.zeros(self.num_classes, device=self.device)
    
    def update(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, H, W) - ground truth labels
        """
        pred_classes = torch.argmax(pred, dim=1)
        
        for c in range(self.num_classes):
            pred_c = (pred_classes == c)
            target_c = (target == c)
            
            self.tp[c] += (pred_c & target_c).sum()
            self.fp[c] += (pred_c & ~target_c).sum()
            self.fn[c] += (~pred_c & target_c).sum()
            self.tn[c] += (~pred_c & ~target_c).sum()
    
    def compute(self):
        """Compute final metrics"""
        epsilon = 1e-7
        
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + epsilon)
        
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        precision = self.tp / (self.tp + self.fp + epsilon)
        
        recall = self.tp / (self.tp + self.fn + epsilon)
        
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        metrics = {
            'dice': dice.cpu().numpy(),
            'iou': iou.cpu().numpy(),
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'f1': f1.cpu().numpy()
        }
        
        return metrics
    
    def get_mean_metrics(self, exclude_background=True):
        """Get mean metrics across classes"""
        metrics = self.compute()
        mean_metrics = {}
        
        start_idx = 1 if exclude_background else 0
        
        for key, values in metrics.items():
            mean_metrics[f'mean_{key}'] = np.mean(values[start_idx:])
            for i in range(len(values)):
                mean_metrics[f'{key}_class_{i}'] = values[i]
        
        return mean_metrics


def compute_dice_score(pred, target, num_classes):
    """
    Compute Dice score
    
    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, H, W) - ground truth labels
        num_classes: number of classes
    
    Returns:
        dice_scores: numpy array of shape (num_classes,)
    """
    pred_classes = torch.argmax(pred, dim=1)
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_scores.append(dice.item())
    
    return np.array(dice_scores)


def compute_iou(pred, target, num_classes):
    """
    Compute IoU (Intersection over Union)
    
    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, H, W) - ground truth labels
        num_classes: number of classes
    
    Returns:
        iou_scores: numpy array of shape (num_classes,)
    """
    pred_classes = torch.argmax(pred, dim=1)
    
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_scores.append(iou.item())
    
    return np.array(iou_scores)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient between prediction and target
    
    Args:
        pred: (H, W) - binary prediction
        target: (H, W) - binary target
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-5):
    """
    Calculate IoU score between prediction and target
    
    Args:
        pred: (H, W) - binary prediction
        target: (H, W) - binary target
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)
