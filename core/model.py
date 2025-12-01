"""
Model handler for breast cancer histopathology classification.
Uses EfficientNet-B0 with Coordinate Attention mechanism.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import io
from typing import Dict, Any, Optional

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FastCoordinateAttention(nn.Module):
    """Coordinate Attention mechanism for feature enhancement"""
    def __init__(self, inp, reduction=16):
        super(FastCoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        
        self.last_attention = None
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        self.last_attention = a_h * a_w
        
        out = identity * self.last_attention
        return out

class FastHistopathologyModel(nn.Module):
    """EfficientNet-B0 based model with Coordinate Attention for histopathology classification"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(FastHistopathologyModel, self).__init__()
        
        # Use EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Feature extraction
        self.features = self.base_model.features
        
        # Efficient attention mechanism
        self.attention = FastCoordinateAttention(inp=1280)  # B0 has 1280 features
        
        # Single pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ModelHandler:
    """Handles model loading and inference for histopathology classification."""
    
    def __init__(self, model_path: str):
        self.device = DEVICE
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.class_names = ['Benign', 'Malignant']

    def _get_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path: str) -> Optional[FastHistopathologyModel]:
        """Load the trained model weights."""
        model = FastHistopathologyModel(num_classes=2, dropout_rate=0.3)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                print(f"✓ Model loaded from {model_path}")
                return model
            except Exception as e:
                print(f"✗ Error loading weights: {e}")
                return None
        else:
            print(f"✗ Model file not found: {model_path}")
            return None

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze a histopathology image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results or error
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get attention metrics
                att_metrics = {}
                if hasattr(self.model.attention, 'last_attention') and self.model.attention.last_attention is not None:
                    att_map = self.model.attention.last_attention.mean(dim=1).cpu()
                    att_metrics['mean_intensity'] = att_map.mean().item()
                    att_metrics['high_attention_ratio'] = (att_map > 0.5).float().mean().item() * 100
                else:
                    att_metrics['mean_intensity'] = 0.0
                    att_metrics['high_attention_ratio'] = 0.0

            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            benign_prob = probabilities[0][0].item()
            malignant_prob = probabilities[0][1].item()

            return {
                "prediction": predicted_class,
                "confidence": confidence_score,
                "benign_probability": benign_prob,
                "malignant_probability": malignant_prob,
                "attention_metrics": att_metrics
            }
        except Exception as e:
            return {"error": str(e)}
