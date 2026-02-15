#!/usr/bin/env python3
"""
SegFormer Model Evaluation Script
Usage: python evaluate_model.py --model_path /path/to/model --test_dir /path/to/test/images
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================================
# Class Mapping
# ============================================================================

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

ID2LABEL = {
    0: "background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
    4: "Dry Bushes", 5: "Ground Clutter", 6: "Logs", 7: "Rocks",
    8: "Landscape", 9: "Sky"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}
N_CLASSES = len(VALUE_MAP)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in VALUE_MAP.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


# ============================================================================
# Dataset Class
# ============================================================================

class EvaluationDataset(Dataset):
    """Simple dataset for evaluation."""
    
    def __init__(self, img_dir: Path, mask_dir: Path, image_processor):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_processor = image_processor
        self.ids = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        if not self.ids:
            raise ValueError(f"No PNG images found in {img_dir}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        image = Image.open(self.img_dir / img_id).convert("RGB")
        mask = Image.open(self.mask_dir / img_id)
        mask = convert_mask(mask)
        
        encoded = self.image_processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoded['pixel_values'].squeeze(0),
            'labels': torch.tensor(mask, dtype=torch.long),
        }


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(
    model_path: str,
    img_dir: str,
    mask_dir: str,
    batch_size: int = 16,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model on test images.
    
    Args:
        model_path: Path to saved model directory
        img_dir: Path to test images
        mask_dir: Path to test masks
        batch_size: Batch size for evaluation
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with metrics
    """
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        
        device = 'cpu'
    
    device = torch.device(device)
    print(f"📍 Device: {device}")
    
    # Load model and processor
    print(f"\n📥 Loading model from {model_path}...")
    try:
        image_processor = SegformerImageProcessor.from_pretrained(model_path)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create dataset
    print(f"\n📁 Loading test data...")
    try:
        dataset = EvaluationDataset(
            Path(img_dir),
            Path(mask_dir),
            image_processor
        )
        print(f"   ✓ Found {len(dataset)} test samples")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # ========================================================================
    # Evaluation Loop
    # ========================================================================
    print(f"\n🔍 Evaluating on {len(dataset)} samples...")
    
    # Accumulators
    total_intersection = torch.zeros(N_CLASSES, device=device)
    total_union = torch.zeros(N_CLASSES, device=device)
    total_pred_area = torch.zeros(N_CLASSES, device=device)
    total_target_area = torch.zeros(N_CLASSES, device=device)
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=True):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample logits to match label size
            upsampled_logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            
            preds = upsampled_logits.argmax(dim=1)
            
            # Pixel accuracy
            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()
            
            # Per-class IoU
            for c in range(N_CLASSES):
                pred_mask = (preds == c)
                target_mask = (labels == c)
                
                total_intersection[c] += (pred_mask & target_mask).sum()
                total_union[c] += (pred_mask | target_mask).sum()
                total_pred_area[c] += pred_mask.sum()
                total_target_area[c] += target_mask.sum()
    
    # ========================================================================
    # Calculate Metrics
    # ========================================================================
    
    class_iou = []
    for c in range(N_CLASSES):
        if total_union[c] > 0:
            iou = (total_intersection[c] / total_union[c]).item()
            class_iou.append(iou)
        else:
            class_iou.append(0.0)
    
    # Mean IoU (only counting classes that exist in ground truth)
    valid_ious = [iou for i, iou in enumerate(class_iou) if total_target_area[i] > 0]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    # Pixel accuracy
    pixel_accuracy = total_correct / total_pixels
    
    # Dice score
    dice_scores = []
    for c in range(N_CLASSES):
        total_area = total_pred_area[c] + total_target_area[c]
        if total_area > 0:
            dice = (2.0 * total_intersection[c] / total_area).item()
            dice_scores.append(dice)
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    # ========================================================================
    # Results
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\n📊 Overall Metrics:")
    print(f"   Mean IoU (mIoU):    {mean_iou:.4f}")
    print(f"   Pixel Accuracy:     {pixel_accuracy:.4f}")
    print(f"   Mean Dice Score:    {mean_dice:.4f}")
    
    print(f"\n📈 Per-Class IoU:")
    print("-" * 50)
    for i in range(N_CLASSES):
        class_name = ID2LABEL[i]
        iou_val = class_iou[i]
        # Mark classes that exist in ground truth
        marker = "✓" if total_target_area[i] > 0 else "✗"
        print(f"   {marker} Class {i:2d} ({class_name:20s}): {iou_val:.4f}")
    print("-" * 50)
    print("="*80)
    
    metrics = {
        'mean_iou': float(mean_iou),
        'pixel_accuracy': float(pixel_accuracy),
        'mean_dice': float(mean_dice),
        'class_iou': class_iou,
        'num_samples': len(dataset),
    }
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SegFormer model on test images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python evaluate_model.py --model_path ./best_test_model \\
                           --test_img_dir ./test/images \\
                           --test_mask_dir ./test/masks
  
  python evaluate_model.py -m ./final_model -i ./test/Color_Images -k ./test/Segmentation
        '''
    )
    
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        required=True,
        help='Path to saved SegFormer model directory'
    )
    
    parser.add_argument(
        '--test_img_dir',
        '-i',
        type=str,
        required=True,
        help='Path to test images folder'
    )
    
    parser.add_argument(
        '--test_mask_dir',
        '-k',
        type=str,
        required=True,
        help='Path to test masks folder'
    )
    
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model_path).exists():
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.test_img_dir).exists():
        print(f"❌ Test image directory not found: {args.test_img_dir}")
        sys.exit(1)
    
    if not Path(args.test_mask_dir).exists():
        print(f"❌ Test mask directory not found: {args.test_mask_dir}")
        sys.exit(1)
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model_path,
        img_dir=args.test_img_dir,
        mask_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()