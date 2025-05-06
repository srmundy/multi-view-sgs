import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class MultiViewSurveillanceDataset(Dataset):
    """
    Dataset class for multi-view surveillance data
    Handles both public UHCTD dataset and custom recorded scenarios
    """
    def __init__(self, 
                 data_root: str,
                 annotation_file: str,
                 temporal_length: int = 16,
                 spatial_size: Tuple[int, int] = (224, 224),
                 mode: str = 'train',
                 transform=None):
        """
        Initialize the dataset
        Args:
            data_root: Root directory containing video files
            annotation_file: Path to CSV file with annotations
            temporal_length: Number of frames to sample from each video
            spatial_size: Spatial dimensions to resize frames to
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply to frames
        """
        self.data_root = Path(data_root)
        self.annotations = pd.read_csv(annotation_file)
        self.temporal_length = temporal_length
        self.spatial_size = spatial_size
        self.mode = mode
        
        # If transform not provided, use default
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Filter annotations by mode (train/val/test)
        if 'split' in self.annotations.columns:
            self.annotations = self.annotations[self.annotations['split'] == mode]
        else:
            # If split not provided, create train/val/test split
            train_val_annotations, test_annotations = train_test_split(
                self.annotations, test_size=0.2, random_state=42, stratify=self.annotations['label'])
            
            train_annotations, val_annotations = train_test_split(
                train_val_annotations, test_size=0.25, random_state=42, stratify=train_val_annotations['label'])
            
            if mode == 'train':
                self.annotations = train_annotations
            elif mode == 'val':
                self.annotations = val_annotations
            else:  # test
                self.annotations = test_annotations
                
        print(f"Loaded {len(self.annotations)} samples for {mode}")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Returns:
            videos: List of tensors, one for each view
            label: Attack label (0: no attack, 1: attack)
        """
        # Get annotation row
        row = self.annotations.iloc[idx]
        
        # Get scene ID and camera views
        scene_id = row['scene_id']
        label = row['label']
        
        # Load video frames from each view
        videos = []
        for view_idx in range(1, row['num_views'] + 1):
            # Construct video path
            video_path = self.data_root / f"scene_{scene_id}" / f"view_{view_idx}.mp4"
            
            # Load frames
            frames = self._load_video_frames(video_path, row['start_frame'], row['end_frame'])
            videos.append(frames)
        
        # Convert to tensor
        videos = [torch.stack([self.transform(frame) for frame in video]) for video in videos]
        
        # Transpose from [frames, channels, height, width] to [channels, frames, height, width]
        videos = [video.permute(1, 0, 2, 3) for video in videos]
        
        return videos, torch.tensor(label, dtype=torch.long)
    
    def _load_video_frames(self, video_path, start_frame, end_frame):
        """
        Load frames from a video file
        Args:
            video_path: Path to video file
            start_frame: First frame to load
            end_frame: Last frame to load
        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
        
        frames = []
        frame_count = 0
        
        # Read frames until we reach the start frame
        while frame_count < start_frame:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        
        # Read frames from start_frame to end_frame
        while frame_count <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, self.spatial_size)
            
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        # If we don't have enough frames, duplicate the last frame
        if len(frames) < self.temporal_length:
            last_frame = frames[-1] if frames else np.zeros((self.spatial_size[0], self.spatial_size[1], 3), dtype=np.uint8)
            while len(frames) < self.temporal_length:
                frames.append(last_frame.copy())
        
        # If we have too many frames, sample evenly
        if len(frames) > self.temporal_length:
            indices = np.linspace(0, len(frames) - 1, self.temporal_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames

def collate_fn(batch):
    """
    Custom collate function for variable number of views
    Args:
        batch: List of (videos, label) tuples
    Returns:
        List of videos and labels
    """
    videos, labels = zip(*batch)
    
    # Determine max number of views
    max_views = max(len(video_list) for video_list in videos)
    
    # Pad videos with zeros if a sample has fewer views
    padded_videos = []
    for video_list in videos:
        if len(video_list) < max_views:
            # Create dummy views with zeros
            video_shape = video_list[0].shape
            for _ in range(max_views - len(video_list)):
                dummy_view = torch.zeros_like(video_list[0])
                video_list.append(dummy_view)
        padded_videos.append(video_list)
    
    # Transpose list of lists to get list of views
    # Each inner list contains the same view from all samples
    views = []
    for view_idx in range(max_views):
        view_tensors = [sample[view_idx] for sample in padded_videos]
        views.append(torch.stack(view_tensors))
    
    labels = torch.stack(labels)
    
    return views, labels

def create_data_loaders(
    data_root: str,
    annotation_file: str,
    batch_size: int = 8,
    temporal_length: int = 16,
    spatial_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4
):
    """
    Create data loaders for training, validation, and testing
    Args:
        data_root: Root directory containing video files
        annotation_file: Path to CSV file with annotations
        batch_size: Batch size for data loaders
        temporal_length: Number of frames to sample from each video
        spatial_size: Spatial dimensions to resize frames to
        num_workers: Number of workers for data loading
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(spatial_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(spatial_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiViewSurveillanceDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        temporal_length=temporal_length,
        spatial_size=spatial_size,
        mode='train',
        transform=train_transform
    )
    
    val_dataset = MultiViewSurveillanceDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        temporal_length=temporal_length,
        spatial_size=spatial_size,
        mode='val',
        transform=val_transform
    )
    
    test_dataset = MultiViewSurveillanceDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        temporal_length=temporal_length,
        spatial_size=spatial_size,
        mode='test',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    
