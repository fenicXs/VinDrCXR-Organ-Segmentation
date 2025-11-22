"""
Training script for Swin-UNet segmentation model
"""

import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # Updated import for PyTorch 2.0+
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.swin_unet import build_swin_unet
from models.swin_unet_pretrained import build_pretrained_swin_unet
from utils.dataset import get_dataloaders
from utils.losses import get_loss_function, SegmentationMetrics
from utils.visualization import visualize_batch, plot_training_curves


class Trainer:
    """Trainer class for segmentation model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.exp_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
        self.output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Building model...")
        if getattr(config, 'USE_PRETRAINED', False):
            print("Using pretrained Swin-UNet model")
            self.model = build_pretrained_swin_unet(config)
        else:
            print("Using custom Swin-UNet model")
            self.model = build_swin_unet(config.SWIN_CONFIG)
        self.model = self.model.to(self.device)
        
        if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Number of parameters: {num_params:.2f}M")
        
        self.criterion = get_loss_function(config)
        self.criterion = self.criterion.to(self.device)
        
        if config.OPTIMIZER == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        self.scheduler = None
        if config.USE_SCHEDULER:
            if config.SCHEDULER_TYPE == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.NUM_EPOCHS,
                    eta_min=1e-6
                )
            elif config.SCHEDULER_TYPE == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=config.SCHEDULER_FACTOR,
                    patience=config.SCHEDULER_PATIENCE,
                    verbose=True
                )
            elif config.SCHEDULER_TYPE == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=config.NUM_EPOCHS // 3,
                    gamma=0.1
                )
        
        self.scaler = GradScaler('cuda') if config.USE_AMP else None
        
        self.grad_accum_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.grad_clip_norm = getattr(config, 'GRADIENT_CLIP_NORM', None)
        self.warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
        
        self.use_pretrained = getattr(config, 'USE_PRETRAINED', False)
        self.freeze_encoder_epochs = getattr(config, 'FREEZE_ENCODER_EPOCHS', 0)
        
        print("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }
        
        self.best_val_dice = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        self.last_vis_data = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.device)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")
        
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.LEARNING_RATE * lr_scale
        
        if self.use_pretrained and epoch == self.freeze_encoder_epochs and hasattr(self.model, 'unfreeze_encoder'):
            print(f"Unfreezing encoder at epoch {epoch + 1}")
            self.model.unfreeze_encoder()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            if self.config.USE_AMP:
                with autocast('cuda'):  # Specify device for PyTorch 2.0+
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.grad_accum_steps
            metrics.update(outputs.detach(), masks)
            
            if batch_idx % self.config.LOG_FREQ == 0:
                pbar.set_postfix({'loss': loss.item() * self.grad_accum_steps})
        
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_metrics = metrics.get_mean_metrics()
        
        return avg_loss, epoch_metrics
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        epoch_loss = 0.0
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.device)
        
        vis_images = []
        vis_masks = []
        vis_preds = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.config.USE_AMP:
                    with autocast('cuda'):  # Specify device for PyTorch 2.0+
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                epoch_loss += loss.item()
                metrics.update(outputs, masks)
                
                if batch_idx == 0:
                    vis_images = images[:4].cpu()
                    vis_masks = masks[:4].cpu()
                    vis_preds = torch.argmax(outputs[:4], dim=1).cpu()
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.val_loader)
        epoch_metrics = metrics.get_mean_metrics()
        
        self.last_vis_data = {
            'images': vis_images,
            'masks': vis_masks, 
            'preds': vis_preds,
            'epoch': epoch
        }
        
        if self.config.SAVE_VISUALIZATIONS and epoch % self.config.VIS_FREQ == 0:
            vis_path = os.path.join(self.output_dir, f'epoch_{epoch+1}_predictions.png')
            visualize_batch(vis_images, vis_masks, vis_preds, num_samples=4, save_path=vis_path)
            print(f"📊 Saved predictions visualization: {vis_path}")
        
        return avg_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'history': self.history,
            'config': self.config.SWIN_CONFIG
        }
        
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.exp_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with Dice: {self.best_val_dice:.4f}")
        
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the N most recent checkpoints"""
        checkpoints = [f for f in os.listdir(self.exp_dir) if f.startswith('checkpoint_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        if len(checkpoints) > self.config.KEEP_BEST_N:
            for ckpt in checkpoints[:-self.config.KEEP_BEST_N]:
                os.remove(os.path.join(self.exp_dir, ckpt))
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        print(f"Experiment: {self.config.EXPERIMENT_NAME}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()
            
            train_loss, train_metrics = self.train_epoch(epoch)
            
            if (epoch + 1) % self.config.VAL_FREQ == 0:
                val_loss, val_metrics = self.validate(epoch)
            else:
                val_loss, val_metrics = None, None
            
            if self.scheduler is not None:
                if self.config.SCHEDULER_TYPE == 'plateau':
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_metrics['mean_dice'])
            self.history['train_iou'].append(train_metrics['mean_iou'])
            self.history['lr'].append(current_lr)
            
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_dice'].append(val_metrics['mean_dice'])
                self.history['val_iou'].append(val_metrics['mean_iou'])
            
            plot_path = os.path.join(self.output_dir, 'training_curves.png')
            try:
                plot_training_curves(self.history, save_path=plot_path)
            except Exception as e:
                print(f"Warning: Could not save training curves: {e}")
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Dice: {train_metrics['mean_dice']:.4f} | IoU: {train_metrics['mean_iou']:.4f}")
            if val_loss is not None:
                print(f"  Val   Loss: {val_loss:.4f} | Dice: {val_metrics['mean_dice']:.4f} | IoU: {val_metrics['mean_iou']:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            is_best = False
            if val_metrics is not None and val_metrics['mean_dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['mean_dice']
                self.best_epoch = epoch
                is_best = True
                self.epochs_without_improvement = 0
                
                if self.config.SAVE_VISUALIZATIONS and self.config.VIS_ON_IMPROVEMENT and hasattr(self, 'last_vis_data'):
                    vis_path = os.path.join(self.output_dir, f'epoch_{epoch+1}_BEST_predictions.png')
                    visualize_batch(
                        self.last_vis_data['images'], 
                        self.last_vis_data['masks'], 
                        self.last_vis_data['preds'], 
                        num_samples=4, 
                        save_path=vis_path
                    )
                    print(f"🏆 NEW BEST MODEL! Saved predictions: {vis_path}")
            else:
                self.epochs_without_improvement += 1
            
            if (epoch + 1) % self.config.SAVE_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            if self.config.USE_EARLY_STOPPING and self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation Dice: {self.best_val_dice:.4f} at epoch {self.best_epoch+1}")
                break
        
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"Training complete! Total time: {total_time/3600:.2f} hours")
        print(f"Best validation Dice: {self.best_val_dice:.4f} at epoch {self.best_epoch+1}")
        print(f"Training curves saved to {os.path.join(self.output_dir, 'training_curves.png')}")


def main():
    parser = argparse.ArgumentParser(description='Train Swin-UNet for chest X-ray segmentation')
    parser.add_argument('--config', type=str, default='config', help='Config module name')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    trainer = Trainer(config)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.history = checkpoint['history']
        trainer.best_val_dice = checkpoint['best_val_dice']
    
    trainer.train()


if __name__ == '__main__':
    main()
