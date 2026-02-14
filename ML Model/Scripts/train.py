"""
Training and validation script for window-based autoencoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import traceback
import gc
import sys
import pickle
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import model utilities
from model import (
    Autoencoder, WindowTickDataDataset, filter_time_windows, 
    prepare_tick_features, prepare_window_data, 
    split_window_data, scale_window_datasets, compute_reconstruction_errors_vectorized
)

# Import MLflow utilities
from mlflow_utils import MLflowPhaseTracker

def main_train_val(params: Dict, paths: Dict[str, Path]):
    """Main training and validation function"""
    print("="*60)
    print("WINDOW-BASED AUTOENCODER TRAINING & VALIDATION")
    print("="*60)
    
    # Extract paths
    artifacts_path = paths['artifacts']
    data_path = paths['data']
    symbols_path = paths['symbols']
    scripts_path = paths['scripts']
    
    # Add scripts path for data_loader import
    sys.path.append(str(scripts_path))
    
    try:
        from data_loader import SimpleTickLoader
    except ImportError as e:
        print(f"❌ Cannot import data_loader: {e}")
        raise
    
    # Create artifacts directory
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    print("📋 Configuration:")
    print(f"  • Artifacts: {artifacts_path}")
    print(f"  • Data: {data_path}")
    print(f"  • Symbols: {symbols_path}")
    print(f"  • MLflow: {'Enabled' if params['mlflow']['enabled'] else 'Disabled'}")
    print(f"  • Epochs: {params['training']['num_epochs']}")
    print(f"  • Batch size: {params['training']['batch_size']}")
    print(f"  • Window size: {params['model'].get('ticks_per_window', 600)} ticks")
    print("="*60)
    
    # Load symbols
    print("Loading symbols...")
    try:
        loader = SimpleTickLoader(str(symbols_path), str(data_path))
        symbols = loader.symbols
        print(f"✅ Loaded {len(symbols)} symbols: {symbols}")
    except Exception as e:
        print(f"❌ Error loading symbols: {e}")
        print("⚠ Using default symbols...")
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        print(f"⚠ Using {len(symbols)} default symbols: {symbols}")
    
    # Process each symbol
    successful_symbols = []
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[Symbol {i}/{len(symbols)}] {symbol}")
        print("-" * 40)
        
        try:
            # Load data for this symbol
            df_symbol = loader.load_one(symbol)
            
            if df_symbol.empty:
                print(f"  No data for {symbol}, skipping...")
                failed_symbols.append(symbol)
                continue
            
            print(f"  Loaded {len(df_symbol):,} ticks")
            
            # Process symbol
            success = process_symbol_train_val(symbol, df_symbol, params, artifacts_path)
            
            if success:
                successful_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
            
            # Clear memory
            del df_symbol
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Save summary
    summary = {
        'total_symbols': len(symbols),
        'successful_symbols': successful_symbols,
        'failed_symbols': failed_symbols,
        'success_count': len(successful_symbols),
        'failed_count': len(failed_symbols),
        'success_rate': f"{(len(successful_symbols)/len(symbols)*100):.1f}%" if symbols else "0%",
        'timestamp': datetime.now().isoformat(),
        'mlflow_enabled': params['mlflow']['enabled'],
        'parameters': params
    }
    
    summary_path = artifacts_path / 'train_val_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING & VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total symbols: {len(symbols)}")
    print(f"✅ Successful: {len(successful_symbols)}")
    print(f"❌ Failed: {len(failed_symbols)}")
    
    if successful_symbols:
        print(f"\n✅ Trained symbols: {', '.join(successful_symbols)}")
    
    print(f"\n📁 Artifacts saved to: {artifacts_path}")
    print(f"📊 Summary: {summary_path}")
    print(f"{'='*60}")
    
    return summary

def process_symbol_train_val(symbol: str, df_symbol: pd.DataFrame, params: Dict, artifacts_path: Path) -> bool:
    """Process a single symbol for training and validation"""
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize MLflow tracker if enabled
        mlflow_tracker = None
        if params['mlflow']['enabled']:
            tracking_uri = params['mlflow'].get('tracking_uri')
            experiment_prefix = params['mlflow'].get('experiment_name_prefix', 'TickAnomaly')
            mlflow_tracker = MLflowPhaseTracker(symbol, experiment_prefix, tracking_uri)
        
        # Initialize trainer
        trainer = SymbolTrainer(symbol, df_symbol, params, artifacts_path, mlflow_tracker)
        
        # Prepare data
        if not trainer.prepare_data():
            print(f"  ❌ Failed to prepare data for {symbol}")
            return False
        
        # Create model
        trainer.create_model()
        
        # Run training
        trainer.run_training_phase()
        
        # Run validation
        trainer.run_validation_phase()
        
        # Save artifacts
        trainer.save_artifacts()
        
        print(f"\n  ✅ {symbol} completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {symbol}: {str(e)}")
        traceback.print_exc()
        return False

class SymbolTrainer:
    """Handles training and validation for a symbol"""
    
    def __init__(self, symbol: str, df_symbol: pd.DataFrame, params: Dict, 
                 artifacts_path: Path, mlflow_tracker = None):
        self.symbol = symbol
        self.df_symbol = df_symbol
        self.params = params
        self.artifacts_path = artifacts_path / symbol
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracker = mlflow_tracker
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        # Training results
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.available_features = None
        self.scaler = None
        self.window_info = None
    
    def prepare_data(self) -> bool:
        """Prepare window-based data"""
        print(f"  Preparing window-based data...")
        
        # Filter time windows
        df_filtered = filter_time_windows(self.df_symbol)
        if len(df_filtered) == 0:
            print(f"  ❌ No data after time filtering")
            return False
        
        # Extract date
        df_filtered['Date'] = df_filtered['DateTime'].dt.date
        unique_dates = sorted(df_filtered['Date'].unique())
        
        # Keep percentage
        keep_percentage = self.params['data']['keep_first_percentage']
        keep_count = int(len(unique_dates) * keep_percentage)
        keep_dates = unique_dates[:keep_count]
        
        print(f"  Total days: {len(unique_dates)}")
        print(f"  Keeping first {keep_count} days ({keep_percentage*100}%)")
        
        df_filtered = df_filtered[df_filtered['Date'].isin(keep_dates)].copy()
        
        if len(df_filtered) == 0:
            print(f"  ❌ No data after date filtering")
            return False
        
        # Prepare features
        df_filtered, self.available_features = prepare_tick_features(df_filtered)
        if not self.available_features:
            print(f"  ❌ No features available")
            return False
        
        print(f"  Features: {self.available_features}")
        
        # Get window size from params or default to 600
        ticks_per_window = self.params['model'].get('ticks_per_window', 600)
        
        # Prepare window-based data (each window separate)
        window_data = prepare_window_data(df_filtered, self.available_features, ticks_per_window)
        if not window_data:
            print(f"  ❌ No window data")
            return False
        
        print(f"  Windows in kept data: {len(window_data)}")
        print(f"  Example window IDs: {list(window_data.keys())[:3]}")
        
        # Split data
        train_ratio = self.params['data']['train_ratio']
        val_ratio = self.params['data']['val_ratio']
        test_ratio = self.params['data']['test_ratio']
        
        self.train_data, self.val_data, self.test_data, self.window_info = split_window_data(
            window_data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
        
        print(f"  Split: Train={len(self.train_data)} windows, Val={len(self.val_data)} windows, Test={len(self.test_data)} windows")
        
        # Scale datasets
        self.train_data_scaled, self.val_data_scaled, self.test_data_scaled, self.scaler = scale_window_datasets(
            self.train_data, self.val_data, self.test_data, self.available_features
        )
        
        # Create datasets
        self.train_dataset = WindowTickDataDataset(self.train_data_scaled, self.available_features, ticks_per_window)
        self.val_dataset = WindowTickDataDataset(self.val_data_scaled, self.available_features, ticks_per_window)
        self.test_dataset = WindowTickDataDataset(self.test_data_scaled, self.available_features, ticks_per_window)
        
        # Create data loaders
        batch_size = min(self.params['training']['batch_size'], len(self.train_dataset))
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        return True
    
    def create_model(self) -> nn.Module:
        """Create window-based model"""
        input_dim = len(self.available_features)
        model_params = self.params['model']
        
        # Get window size from params or default to 600
        ticks_per_window = model_params.get('ticks_per_window', 600)
        
        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dims=model_params['hidden_dims'],
            latent_dim=model_params['latent_dim'],
            ticks_per_window=ticks_per_window,
            dropout_rate=model_params['dropout_rate'],
            activation=model_params['activation'],
            use_batch_norm=model_params['use_batch_norm'],
            weight_decay=model_params['weight_decay']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Model: {ticks_per_window} × {input_dim} → {model_params['hidden_dims']} → {model_params['latent_dim']}")
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return self.model
    
    def run_training_phase(self):
        """Run training"""
        print(f"\n  Training...")
        
        mlflow_enabled = self.params['mlflow']['enabled']
        if mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.start_phase_run("training")
            
            # Log parameters
            training_params = {
                'symbol': self.symbol,
                'epochs': self.params['training']['num_epochs'],
                'batch_size': self.params['training']['batch_size'],
                'learning_rate': self.params['training']['learning_rate'],
                'window_size': self.params['model'].get('ticks_per_window', 600),
                'train_windows': len(self.train_dataset),
                'val_windows': len(self.val_dataset)
            }
            self.mlflow_tracker.log_phase_params(training_params)
        
        criterion = nn.MSELoss()
        training_params = self.params['training']
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params['weight_decay']
        )
        
        # Training loop
        num_epochs = training_params['num_epochs']
        early_stopping_enabled = training_params['early_stopping'].lower() == 'yes'
        patience = training_params['patience']
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch in self.train_loader:
                batch = batch.to(self.device)
                reconstructed, _ = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_train_loss += loss.item() * batch.size(0)
            
            avg_train_loss = epoch_train_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = batch.to(self.device)
                    reconstructed, _ = self.model(batch)
                    loss = criterion(reconstructed, batch)
                    epoch_val_loss += loss.item() * batch.size(0)
            
            avg_val_loss = epoch_val_loss / len(self.val_loader.dataset)
            self.val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                torch.save(self.model.state_dict(), self.artifacts_path / 'best_model.pth')
            else:
                patience_counter += 1
                if early_stopping_enabled and patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 1 == 0:
                print(f'    Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        if mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.log_phase_metrics({
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'best_val_loss': self.best_val_loss
            })
            self.mlflow_tracker.end_phase_run()
        
        print(f"  Training completed: {len(self.train_losses)} epochs, Best Val Loss: {self.best_val_loss:.6f}")
    
    def run_validation_phase(self):
        """Run validation"""
        print(f"\n  Validation...")
        
        mlflow_enabled = self.params['mlflow']['enabled']
        if mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.start_phase_run("validation")
        
        self.model.eval()
        criterion = nn.MSELoss()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                reconstructed, _ = self.model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item() * batch.size(0)
        
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        print(f"  Validation Loss: {avg_val_loss:.6f}")
        
        if mlflow_enabled and self.mlflow_tracker:
            self.mlflow_tracker.log_phase_metrics({
                'validation_loss': avg_val_loss
            })
            self.mlflow_tracker.end_phase_run()
    
    def save_artifacts(self):
        """Save all training artifacts"""
        print(f"  Saving artifacts...")
        
        # Get window size from model
        ticks_per_window = self.model.ticks_per_window
        
        # 1. Save model configuration
        config_path = self.artifacts_path / 'model_config.json'
        config = {
            'symbol': self.symbol,
            'model_params': self.params['model'],
            'training_params': self.params['training'],
            'features': self.available_features,
            'input_dim': len(self.available_features),
            'ticks_per_window': ticks_per_window,
            'device': str(self.device),
            'window_info': self.window_info,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_summary': {
                'best_val_loss': self.best_val_loss,
                'total_epochs': len(self.train_losses),
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'train_windows': len(self.train_dataset),
                'val_windows': len(self.val_dataset),
                'test_windows': len(self.test_dataset)
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✅ Model config saved: {config_path}")
        
        # 2. Save trained model
        model_path = self.artifacts_path / 'trained_autoencoder.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': config,
            'ticks_per_window': ticks_per_window
        }, model_path)
        print(f"  ✅ Model saved: {model_path}")
        
        # 3. Save scaler
        scaler_path = self.artifacts_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✅ Scaler saved: {scaler_path}")
        
        # 4. Save test data
        test_data_path = self.artifacts_path / 'test_data.pkl'
        with open(test_data_path, 'wb') as f:
            pickle.dump(self.test_data_scaled, f)
        print(f"  ✅ Test data saved: {test_data_path}")
        
        # 5. Save training curves
        curves_path = self.artifacts_path / 'training_curves.json'
        curves_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': list(range(len(self.train_losses)))
        }
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f, indent=2)
        print(f"  ✅ Training curves saved: {curves_path}")
        
        print(f"  ✅ All artifacts saved to: {self.artifacts_path}")

if __name__ == "__main__":
    print("❌ This script should be run through FastAPI")
    print("   Use: python main.py")
    print("   Then access: http://127.0.0.1:9935/docs")