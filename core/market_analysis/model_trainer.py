
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from datetime import datetime

from .deep_feature_extractor import DeepFeatureExtractor, EnhancedFeatureExtractor

class MarketDataset(Dataset):
    """Custom dataset for market data"""
    def __init__(self, data: pd.DataFrame, window_size: int = 50):
        self.data = data
        self.window_size = window_size
        self.feature_calculator = EnhancedFeatureExtractor(config={})
        
    def __len__(self) -> int:
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data.iloc[idx:idx + self.window_size]
        
        # Calculate input features
        features = self.feature_calculator._prepare_features(window)
        
        # Calculate target values (you can modify these based on your needs)
        next_returns = self.data.iloc[idx + self.window_size]['close'] / \
                      self.data.iloc[idx + self.window_size - 1]['close'] - 1
        volatility = self.data.iloc[idx:idx + self.window_size]['close'].pct_change().std()
        trend = self._calculate_trend_strength(window)
        
        # Create target tensor
        target = torch.tensor([next_returns, volatility, trend], dtype=torch.float32)
        
        return torch.FloatTensor(features), target
        
    def _calculate_trend_strength(self, window: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        prices = window['close'].values
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        r2 = np.corrcoef(x, prices)[0, 1] ** 2
        return float(abs(slope) * r2)

class ModelTrainer:
    """Handles training and evaluation of the deep learning model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepFeatureExtractor(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    async def train(self, 
                    train_data: pd.DataFrame,
                    val_data: pd.DataFrame,
                    epochs: int = 50,
                    batch_size: int = 32) -> Dict:
        """Train the model on historical market data"""
        try:
            # Create datasets
            train_dataset = MarketDataset(train_data)
            val_dataset = MarketDataset(val_data)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rates': []
            }
            
            # Training loop
            for epoch in range(epochs):
                train_loss = await self._train_epoch(train_loader)
                val_loss = await self._validate_epoch(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Log progress
                logging.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('checkpoint_frequency', 5) == 0:
                    self._save_checkpoint(epoch, val_loss)
                    
            return history
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
            
    async def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        criterion = nn.MSELoss()
        
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Calculate loss
            loss = criterion(
                torch.cat([
                    outputs.temporal_patterns,
                    outputs.market_structure.squeeze(1),
                    outputs.regime_indicators
                ], dim=1),
                targets
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    async def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(
                    torch.cat([
                        outputs.temporal_patterns,
                        outputs.market_structure.squeeze(1),
                        outputs.regime_indicators
                    ], dim=1),
                    targets
                )
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}_{datetime.now():%Y%m%d_%H%M}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, checkpoint_path)
        
        logging.info(f"Saved checkpoint to {checkpoint_path}")
