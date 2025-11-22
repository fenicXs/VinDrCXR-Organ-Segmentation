"""
Inference script for Swin-UNet segmentation model
"""

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.swin_unet_pretrained import build_pretrained_swin_unet
from utils.dataset import CheXmaskDataset
from utils.losses import SegmentationMetrics
from utils.visualization import visualize_segmentation, create_comparison_grid, save_prediction_as_mask


class Predictor:
    """Predictor class for segmentation model"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Building model...")
        self.model = build_pretrained_swin_unet(config)
        self.model = self.model.to(self.device)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False  # allow loading checkpoints saved with Pickle globals
        )
        
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for key in attn_mask_keys:
            state_dict.pop(key)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']+1}")
        print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    
    def predict_image(self, image):
        """
        Predict segmentation mask for a single image
        
        Args:
            image: (H, W) or (1, H, W) - grayscale image
            
        Returns:
            pred_mask: (H, W) - predicted class mask
            prob_map: (C, H, W) - probability map for each class
        """
        if len(image.shape) == 2:
            image = image[np.newaxis, :]
        
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        orig_h, orig_w = image.shape[1:]
        image_resized = cv2.resize(image[0], (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        image_resized = image_resized[np.newaxis, np.newaxis, :]
        
        image_tensor = torch.from_numpy(image_resized).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prob_map = F.softmax(output, dim=1)
            pred_mask = torch.argmax(output, dim=1)
        
        prob_map = F.interpolate(
            prob_map,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=(orig_h, orig_w),
            mode='nearest'
        ).squeeze(1).long()
        
        pred_mask = pred_mask.cpu().numpy()[0]
        prob_map = prob_map.cpu().numpy()[0]
        
        return pred_mask, prob_map
    
    def predict_dataset(self, dataset, output_dir, save_masks=True, save_visualizations=True):
        """
        Run predictions on entire dataset
        
        Args:
            dataset: CheXmaskDataset instance
            output_dir: directory to save results
            save_masks: whether to save prediction masks
            save_visualizations: whether to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if save_masks:
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.device)
        
        results = []
        
        print(f"Running inference on {len(dataset)} samples...")
        
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            image = sample['image'].numpy()[0]  # Remove channel dimension
            gt_mask = sample['mask'].numpy()
            image_id = sample['image_id']
            
            pred_mask, prob_map = self.predict_image(image)
            
            with torch.no_grad():
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).to(self.device)
                gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).to(self.device)
                
                pred_one_hot = F.one_hot(pred_tensor, self.config.NUM_CLASSES).permute(0, 3, 1, 2).float()
                
                metrics.update(pred_one_hot, gt_tensor)
            
            sample_metrics = {}
            for c in range(self.config.NUM_CLASSES):
                pred_c = (pred_mask == c)
                gt_c = (gt_mask == c)
                
                tp = np.logical_and(pred_c, gt_c).sum()
                fp = np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                fn = np.logical_and(np.logical_not(pred_c), gt_c).sum()
                
                dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
                iou = tp / (tp + fp + fn + 1e-7)
                
                sample_metrics[f'dice_class_{c}'] = dice
                sample_metrics[f'iou_class_{c}'] = iou
            
            results.append({
                'image_id': image_id,
                'mean_dice': np.mean([sample_metrics[f'dice_class_{c}'] for c in range(1, self.config.NUM_CLASSES)]),
                'mean_iou': np.mean([sample_metrics[f'iou_class_{c}'] for c in range(1, self.config.NUM_CLASSES)]),
                **sample_metrics
            })
            
            if save_masks:
                mask_path = os.path.join(masks_dir, f'{image_id}_pred.png')
                save_prediction_as_mask(pred_mask, mask_path)
            
            if save_visualizations and idx < 50:  # Save first 50 samples
                vis_path = os.path.join(vis_dir, f'{image_id}_comparison.png')
                visualize_segmentation(image, gt_mask, pred_mask, save_path=vis_path)
                
                comp_path = os.path.join(vis_dir, f'{image_id}_detailed.png')
                create_comparison_grid(image, gt_mask, pred_mask, save_path=comp_path)
        
        overall_metrics = metrics.get_mean_metrics()
        
        print("\n" + "="*50)
        print("Overall Metrics:")
        print(f"  Mean Dice: {overall_metrics['mean_dice']:.4f}")
        print(f"  Mean IoU: {overall_metrics['mean_iou']:.4f}")
        print("-"*50)
        
        class_names = ['Background', 'Right Lung', 'Left Lung', 'Heart']
        for c in range(self.config.NUM_CLASSES):
            print(f"  {class_names[c]}:")
            print(f"    Dice: {overall_metrics[f'dice_class_{c}']:.4f}")
            print(f"    IoU: {overall_metrics[f'iou_class_{c}']:.4f}")
        
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, 'predictions_metrics.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        metrics_path = os.path.join(output_dir, 'overall_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Overall Metrics\n")
            f.write("="*50 + "\n")
            for key, value in overall_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Overall metrics saved to {metrics_path}")
        
        return results_df, overall_metrics
    
    def predict_single_image(self, image_path, save_path=None):
        """
        Predict segmentation for a single image file
        
        Args:
            image_path: path to image file
            save_path: path to save visualization (optional)
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        pred_mask, prob_map = self.predict_image(image)
        
        if save_path:
            visualize_segmentation(image / 255.0, pred_mask, None, save_path=save_path)
        
        return pred_mask, prob_map


def main():
    parser = argparse.ArgumentParser(description='Run inference with Swin-UNet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'val', 'test'], 
                       help='Which dataset split to use')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--save_masks', action='store_true', help='Save prediction masks')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--image_path', type=str, default=None, help='Path to single image for prediction')
    args = parser.parse_args()
    
    predictor = Predictor(config, args.checkpoint)
    
    if args.image_path:
        output_path = args.output_dir or 'prediction.png'
        pred_mask, prob_map = predictor.predict_single_image(args.image_path, save_path=output_path)
        print(f"Prediction saved to {output_path}")
        print(f"Predicted classes: {np.unique(pred_mask)}")
        return
    
    from utils.dataset import create_data_splits
    
    use_official_split = (
        getattr(config, "USE_VINDR_OFFICIAL_SPLIT", False)
        and str(config.DATASET_NAME).lower() == "vindr-cxr"
    )
    split_val_ratio = getattr(config, "VAL_RATIO", 0.15)

    train_indices, val_indices, test_indices = create_data_splits(
        config.CSV_PATH,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=split_val_ratio,
        test_ratio=config.TEST_RATIO,
        quality_threshold=config.QUALITY_THRESHOLD,
        seed=config.SEED,
        dataset_name=config.DATASET_NAME,
        train_split_file=getattr(config, "VINDR_TRAIN_SPLIT_PATH", None) if use_official_split else None,
        test_split_file=getattr(config, "VINDR_TEST_SPLIT_PATH", None) if use_official_split else None,
        use_official_split=use_official_split
    )
    
    if args.dataset == 'train':
        indices = train_indices
    elif args.dataset == 'val':
        indices = val_indices
    else:
        indices = test_indices
    
    full_dataset = CheXmaskDataset(
        csv_path=config.CSV_PATH,
        image_base_path=config.IMAGE_BASE_PATH,
        dataset_name=config.DATASET_NAME,
        quality_threshold=config.QUALITY_THRESHOLD,
        transform=None,
        mode=args.dataset
    )
    
    dataset = torch.utils.data.Subset(full_dataset, indices)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME, f'inference_{args.dataset}')
    
    results_df, overall_metrics = predictor.predict_dataset(
        dataset,
        args.output_dir,
        save_masks=args.save_masks,
        save_visualizations=args.save_vis
    )


if __name__ == '__main__':
    main()
