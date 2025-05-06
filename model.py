import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional

class MultiViewFeatureExtractor(nn.Module):
    """
    Extracts spatial-temporal features from multiple camera views
    """
    def __init__(self, 
                 input_channels: int = 3, 
                 hidden_dim: int = 256,
                 temporal_length: int = 16):
        super().__init__()
        
        # Spatial feature extraction using ResNet-like blocks
        self.spatial_features = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # ResNet-like blocks
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, hidden_dim, stride=2)
        )
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(temporal_length//2, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Create a ResNet-like block"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process multiple camera views
        Args:
            x: List of camera view inputs, each of shape [batch_size, channels, frames, height, width]
        Returns:
            List of extracted features for each view
        """
        features = []
        for view in x:
            # Extract spatial-temporal features
            spatial_feats = self.spatial_features(view)
            temporal_feats = self.temporal_conv(spatial_feats)
            # Reshape for transformer input: [batch, seq_len, features]
            b, c, t, h, w = temporal_feats.shape
            view_features = temporal_feats.permute(0, 2, 1, 3, 4).contiguous()
            view_features = view_features.view(b, t, c * h * w)
            features.append(view_features)
            
        return features

class ResNetBlock(nn.Module):
    """Basic ResNet block for feature extraction"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), 
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), 
                               stride=1, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention for Vision Transformer"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.shape[0]
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        
        return output

class CrossViewAttention(nn.Module):
    """Cross-view attention mechanism to correlate information across camera views"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, view_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-view attention between different camera perspectives
        Args:
            view_features: List of features from each camera view
                          [batch_size, seq_len, embed_dim]
        Returns:
            Enhanced features with cross-view information
        """
        enhanced_features = []
        num_views = len(view_features)
        
        for i in range(num_views):
            # Self attention for each view
            x = view_features[i]
            x = x + self.dropout(self.self_attn(x, x, x))
            x = self.norm1(x)
            
            # Cross attention with other views
            cross_x = x.clone()
            for j in range(num_views):
                if i != j:
                    # Apply cross attention with other views
                    context = view_features[j]
                    cross_x = cross_x + self.dropout(self.cross_attn(cross_x, context, context))
            
            cross_x = self.norm2(cross_x)
            enhanced_features.append(cross_x)
            
        return enhanced_features

class ViTEncoderLayer(nn.Module):
    """Vision Transformer Encoder Layer"""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention block
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        # Feed-forward block
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer backbone for spatiotemporal processing"""
    def __init__(self, embed_dim, num_heads, num_layers, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        Process features through transformer layers
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
        Returns:
            Processed features with same shape
        """
        for layer in self.layers:
            x = layer(x)
        return x

class LightweightDiffusion(nn.Module):
    """Lightweight diffusion model for trajectory prediction and uncertainty modeling"""
    def __init__(self, feature_dim, hidden_dim, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        
        # Noise predictor network
        self.noise_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, timesteps=None):
        """
        Forward pass for the diffusion model
        Args:
            x: Input features [batch_size, seq_len, feature_dim]
            timesteps: Optional specific timesteps for diffusion process
        """
        if timesteps is None:
            # Use a single step for inference (simplified)
            timesteps = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        
        # Predict noise/trajectory deviation
        noise_prediction = self.noise_predictor(x)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(x)
        
        return noise_prediction, uncertainty
    
    def sample(self, shape, device):
        """
        Generate trajectory samples using diffusion process
        Args:
            shape: Shape of samples to generate
            device: Device to generate samples on
        Returns:
            Generated trajectory samples
        """
        # Start with random noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process (simplified)
        for t in range(self.num_steps - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device).repeat(shape[0])
            
            # Predict noise
            noise_pred, _ = self.forward(x, t_tensor)
            
            # Update sample based on noise prediction (simplified diffusion step)
            alpha = 1.0 - t / self.num_steps  # Simplified schedule
            x = (x - (1 - alpha) * noise_pred) / alpha.sqrt()
            
            # Add noise scaled by timestep (except at t=0)
            if t > 0:
                noise_scale = (1.0 - alpha).sqrt()
                noise = torch.randn_like(x)
                x = x + noise_scale * noise
                
        return x

class DetectionModule(nn.Module):
    """Final detection module for attack prediction"""
    def __init__(self, feature_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, vit_features, diffusion_features, uncertainty):
        """
        Generate final attack predictions
        Args:
            vit_features: Features from Vision Transformer
            diffusion_features: Features from diffusion model
            uncertainty: Uncertainty estimates from diffusion model
        Returns:
            Attack predictions with confidence scores
        """
        # Combine features from both streams
        combined_features = vit_features + diffusion_features * (1.0 - uncertainty)
        
        # Temporal pooling across sequence length
        pooled_features = torch.mean(combined_features, dim=1)
        
        # Classification
        logits = self.fusion(pooled_features)
        confidence = F.softmax(logits, dim=-1)
        
        return logits, confidence

class CrossingTheStreams(nn.Module):
    """
    Complete architecture for multi-view surveillance attack prediction
    """
    def __init__(self,
                 input_channels: int = 3,
                 num_views: int = 2,
                 temporal_length: int = 16,
                 feature_dim: int = 256,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_transformer_layers: int = 6,
                 ffn_dim: int = 2048,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_views = num_views
        
        # Feature extraction from multiple views
        self.feature_extractor = MultiViewFeatureExtractor(
            input_channels=input_channels,
            hidden_dim=feature_dim,
            temporal_length=temporal_length
        )
        
        # Projection from feature_dim to embed_dim for transformer
        self.projection = nn.Linear(feature_dim * 7 * 7, embed_dim)  # Assuming 7x7 spatial dimensions
        
        # Cross-view attention
        self.cross_view_attention = CrossViewAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Vision Transformer backbone
        self.vision_transformer = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        # Lightweight diffusion component
        self.diffusion = LightweightDiffusion(
            feature_dim=embed_dim,
            hidden_dim=ffn_dim // 2
        )
        
        # Final detection module
        self.detection = DetectionModule(
            feature_dim=embed_dim,
            hidden_dim=ffn_dim // 2,
            num_classes=num_classes
        )
        
    def forward(self, inputs: List[torch.Tensor]):
        """
        Forward pass for the full architecture
        Args:
            inputs: List of tensor inputs from multiple cameras
                   Each tensor shape: [batch_size, channels, frames, height, width]
        Returns:
            Attack predictions with confidence scores
        """
        # Extract features from each view
        view_features = self.feature_extractor(inputs)
        
        # Project features to embedding dimension
        projected_features = []
        for view_feat in view_features:
            projected_features.append(self.projection(view_feat))
        
        # Apply cross-view attention
        enhanced_features = self.cross_view_attention(projected_features)
        
        # Combine multiple views (simplified - using mean pooling)
        combined_features = torch.stack(enhanced_features, dim=0).mean(dim=0)
        
        # Process through Vision Transformer
        vit_features = self.vision_transformer(combined_features)
        
        # Process through lightweight diffusion
        diffusion_pred, uncertainty = self.diffusion(vit_features)
        
        # Final detection
        logits, confidence = self.detection(vit_features, diffusion_pred, uncertainty)
        
        return {
            "logits": logits,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "vit_features": vit_features,
            "diffusion_features": diffusion_pred
        }
    
    def train_step(self, inputs, targets, optimizer, criterion):
        """Training step for the model"""
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = criterion(outputs["logits"], targets)
        loss.backward()
        optimizer.step()
        return loss, outputs

def build_crossing_streams_model(
    num_views: int = 2,
    input_channels: int = 3,
    temporal_length: int = 16,
    feature_dim: int = 256,
    embed_dim: int = 512,
    num_classes: int = 2
) -> CrossingTheStreams:
    """
    Build the Crossing the Streams model with default parameters
    Args:
        num_views: Number of camera views
        input_channels: Number of input channels (3 for RGB)
        temporal_length: Number of frames in temporal sequence
        feature_dim: Feature dimension for spatial features
        embed_dim: Embedding dimension for transformer
        num_classes: Number of output classes (2 for binary attack detection)
    Returns:
        Initialized CrossingTheStreams model
    """
    model = CrossingTheStreams(
        input_channels=input_channels,
        num_views=num_views,
        temporal_length=temporal_length,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=8,
        num_transformer_layers=6,
        ffn_dim=2048,
        num_classes=num_classes,
        dropout=0.1
    )
    return model

# Example usage:
if __name__ == "__main__":
    # Create sample inputs (2 camera views)
    batch_size = 4
    channels = 3
    frames = 16
    height = 224
    width = 224
    num_views = 2
    
    # Example input: List of tensors, one for each camera view
    inputs = [torch.randn(batch_size, channels, frames, height, width) for _ in range(num_views)]
    
    # Create model
    model = build_crossing_streams_model(num_views=num_views)
    
    # Forward pass
    outputs = model(inputs)
    
    # Print output shapes
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"ViT features shape: {outputs['vit_features'].shape}")
    print(f"Diffusion features shape: {outputs['diffusion_features'].shape}")
