#!/usr/bin/env python3
"""
Minimal Working Example for Lightning Multi-GPU Training
This is the simplest possible multi-GPU Lightning script to test SLURM configuration.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(pl.LightningModule):
    """Simplest possible model for testing multi-GPU."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Log GPU usage info
        if batch_idx % 10 == 0:
            device = str(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(f"Step {batch_idx}, Device: {device}, Memory: {allocated:.2f}GB")
        
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def create_dummy_data():
    """Create dummy dataset for testing."""
    # Create random data
    X = torch.randn(1000, 100)
    y = torch.randn(1000, 1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
    return dataloader


def main():
    """Main function for MWE."""
    print("🚀 Lightning Multi-GPU MWE")
    print("=" * 50)
    
    # Print environment info
    print("Environment Info:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    
    # Create model and data
    model = SimpleMLP()
    train_dataloader = create_dummy_data()
    
    # Try different strategies
    device_count = torch.cuda.device_count()
    
    strategies_to_test = []
    if device_count > 1:
        strategies_to_test = [
            ("ddp_spawn", "auto"),
            ("ddp", "auto"), 
            ("auto", "auto")
        ]
    else:
        strategies_to_test = [("auto", 1)]
    
    for strategy_name, devices in strategies_to_test:
        print(f"\n🧪 Testing strategy: {strategy_name}, devices: {devices}")
        print("-" * 30)
        
        try:
            # Create trainer
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=devices,
                strategy=strategy_name,
                max_epochs=1,
                max_steps=20,  # Very short test
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=True,
                log_every_n_steps=5
            )
            
            print(f"✅ Trainer created successfully")
            print(f"   Strategy: {trainer.strategy}")
            print(f"   Devices: {trainer.num_devices}")
            
            # Test training
            print(f"🏃 Starting training...")
            trainer.fit(model, train_dataloader)
            print(f"✅ Training completed successfully!")
            
            # Print GPU memory usage after training
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            print(f"🎉 SUCCESS with {strategy_name}!")
            break  # Stop on first successful strategy
            
        except Exception as e:
            print(f"❌ FAILED with {strategy_name}: {str(e)}")
            # Clear any GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    else:
        print("\n💥 ALL STRATEGIES FAILED!")
        print("This suggests a fundamental SLURM/environment issue.")
    
    print("\n" + "=" * 50)
    print("MWE Complete")


if __name__ == "__main__":
    main()