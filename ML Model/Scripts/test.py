"""
Testing script for window-based autoencoder
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pickle
import traceback
import gc
from typing import Dict

# Import model utilities
from model import (
    Autoencoder, WindowTickDataDataset, compute_reconstruction_errors_vectorized
)

# Import MLflow utilities
from mlflow_utils import MLflowPhaseTracker

def main_test(params: Dict, paths: Dict[str, Path]):
    """Main testing function"""
    print("="*60)
    print("WINDOW-BASED AUTOENCODER TESTING")
    print("="*60)
    
    # Extract paths
    artifacts_path = paths['artifacts']
    
    print(f"📁 Artifacts path: {artifacts_path}")
    print(f"📊 MLflow: {'Enabled' if params['mlflow']['enabled'] else 'Disabled'}")
    print("="*60)
    
    # Find trained symbols
    trained_symbols = []
    for item in artifacts_path.iterdir():
        if item.is_dir():
            model_path = item / 'trained_autoencoder.pth'
            best_model_path = item / 'best_model.pth'
            test_data_path = item / 'test_data.pkl'
            
            if (model_path.exists() or best_model_path.exists()) and test_data_path.exists():
                trained_symbols.append(item.name)
    
    if not trained_symbols:
        print("❌ No trained models found!")
        print("   Please run training first")
        raise ValueError("No trained models found")
    
    print(f"✅ Found {len(trained_symbols)} trained symbol(s): {', '.join(trained_symbols)}")
    
    # Process each symbol
    successful_symbols = []
    failed_symbols = []
    
    for i, symbol in enumerate(trained_symbols, 1):
        print(f"\n[Symbol {i}/{len(trained_symbols)}] {symbol}")
        print("-" * 40)
        
        try:
            success = process_symbol_test(symbol, artifacts_path, params)
            
            if success:
                successful_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {e}")
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Save summary
    summary = {
        'total_symbols': len(trained_symbols),
        'successful_symbols': successful_symbols,
        'failed_symbols': failed_symbols,
        'success_count': len(successful_symbols),
        'failed_count': len(failed_symbols),
        'success_rate': f"{(len(successful_symbols)/len(trained_symbols)*100):.1f}%" if trained_symbols else "0%",
        'timestamp': datetime.now().isoformat(),
        'mlflow_enabled': params['mlflow']['enabled']
    }
    
    summary_path = artifacts_path / 'test_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Total symbols tested: {len(trained_symbols)}")
    print(f"✅ Successful: {len(successful_symbols)}")
    print(f"❌ Failed: {len(failed_symbols)}")
    
    if successful_symbols:
        print(f"\n✅ Tested symbols: {', '.join(successful_symbols)}")
    
    print(f"\n📁 Summary saved: {summary_path}")
    print(f"{'='*60}")
    
    return summary

def process_symbol_test(symbol: str, artifacts_path: Path, params: Dict) -> bool:
    """Test a single trained symbol"""
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
        
        # Initialize tester
        tester = SymbolTester(symbol, artifacts_path, params, mlflow_tracker)
        
        # Load model
        tester.load_model()
        
        # Load test data
        tester.load_test_data()
        
        # Run testing
        tester.run_testing_phase()
        
        print(f"\n  ✅ {symbol} testing completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing {symbol}: {str(e)}")
        traceback.print_exc()
        return False

class SymbolTester:
    """Handles testing of a trained symbol"""
    
    def __init__(self, symbol: str, artifacts_path: Path, params: Dict, mlflow_tracker = None):
        self.symbol = symbol
        self.artifacts_path = artifacts_path / symbol
        self.params = params
        self.mlflow_tracker = mlflow_tracker
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        self.config = None
        self.ticks_per_window = 600
    
    def get_or_create_config(self):
        """Get config from file or create from parameters"""
        config_path = self.artifacts_path / 'model_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"  ✅ Config loaded from {config_path}")
            
            if 'ticks_per_window' in self.config:
                self.ticks_per_window = self.config['ticks_per_window']
        else:
            print(f"  ⚠ Config file not found. Creating from parameters...")
            
            self.config = {
                'model_params': self.params['model'],
                'training_params': self.params['training'],
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'created_from_params': True
            }
            print(f"  ✅ Config created from parameters")
    
    def load_model(self):
        """Load trained model"""
        print(f"  Loading model...")
        
        # Get or create config
        self.get_or_create_config()
        
        # Find model file
        model_path = self.artifacts_path / 'trained_autoencoder.pth'
        if not model_path.exists():
            model_path = self.artifacts_path / 'best_model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found for {self.symbol}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine input dimension
        if self.config and 'features' in self.config:
            input_dim = len(self.config['features'])
        else:
            input_dim = len(self.params['data']['features'])
        
        # Get window size
        if 'ticks_per_window' in checkpoint:
            self.ticks_per_window = checkpoint['ticks_per_window']
        elif 'ticks_per_window' in self.config:
            self.ticks_per_window = self.config['ticks_per_window']
        else:
            self.ticks_per_window = self.params['model'].get('ticks_per_window', 600)
        
        # Create model
        model_params = self.config.get('model_params', self.params['model'])
        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dims=model_params['hidden_dims'],
            latent_dim=model_params['latent_dim'],
            ticks_per_window=self.ticks_per_window,
            dropout_rate=model_params['dropout_rate'],
            activation=model_params['activation'],
            use_batch_norm=model_params['use_batch_norm'],
            weight_decay=model_params['weight_decay']
        ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"  ✅ Model loaded from {model_path}")
        print(f"  Window size: {self.ticks_per_window} ticks")
        
        return self.model
    
    def load_test_data(self):
        """Load test data and scaler"""
        print(f"  Loading test data...")
        
        # Load scaler
        scaler_path = self.artifacts_path / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"No scaler file found for {self.symbol}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load test data
        test_data_path = self.artifacts_path / 'test_data.pkl'
        if not test_data_path.exists():
            raise FileNotFoundError(f"No test data file found for {self.symbol}")
        
        with open(test_data_path, 'rb') as f:
            self.test_data_scaled = pickle.load(f)
        
        # Determine features from test data
        if isinstance(self.test_data_scaled, dict) and len(self.test_data_scaled) > 0:
            first_window = list(self.test_data_scaled.keys())[0]
            first_df = self.test_data_scaled[first_window]
            
            metadata_cols = ['Window_ID', 'Original_Tick_Count']
            features = [col for col in first_df.columns if col not in metadata_cols]
            
            if self.config:
                self.config['features'] = features
                self.config['input_dim'] = len(features)
            else:
                self.config = {
                    'features': features,
                    'input_dim': len(features),
                    'model_params': self.params['model'],
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
            
            print(f"  Features from test data: {features}")
        else:
            features = self.params['data']['features']
            if self.config:
                self.config['features'] = features
                self.config['input_dim'] = len(features)
            else:
                self.config = {
                    'features': features,
                    'input_dim': len(features),
                    'model_params': self.params['model'],
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
            print(f"  Features from parameters: {features}")
        
        # Create dataset and dataloader
        self.test_dataset = WindowTickDataDataset(
            self.test_data_scaled, 
            features, 
            ticks_per_window=self.ticks_per_window
        )
        batch_size = min(8, len(self.test_dataset))
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  Test windows: {len(self.test_dataset)}")
        return self.test_loader
    
    def run_testing_phase(self):
        """Run testing and log to MLflow"""
        print(f"\n  Testing...")
        
        mlflow_enabled = self.params['mlflow']['enabled']
        
        # Start MLflow run if enabled
        if mlflow_enabled and self.mlflow_tracker:
            try:
                self.mlflow_tracker.start_phase_run("testing", tags={"phase": "testing", "model_phase": "final"})
            except Exception as e:
                print(f"    ⚠ Warning: Could not start MLflow run: {e}")
        
        self.model.eval()
        
        # Compute reconstruction errors
        test_errors = compute_reconstruction_errors_vectorized(self.model, self.test_loader, self.device)
        
        if len(test_errors) == 0:
            print("  ⚠ No test errors computed")
            if mlflow_enabled and self.mlflow_tracker:
                self.mlflow_tracker.end_phase_run()
            return
        
        # Convert test_errors to float64 to avoid float32 serialization issues
        test_errors = test_errors.astype(np.float64)
        
        # Calculate anomaly thresholds
        percentiles = self.params['anomaly_detection']['percentile_thresholds']
        thresholds = {}
        for p in percentiles:
            # Convert to Python float to avoid numpy float32
            thresholds[p] = float(np.percentile(test_errors, p))
        
        # Count anomalies
        anomaly_counts = {}
        for p in percentiles:
            # Convert to Python int
            anomaly_counts[p] = int(np.sum(test_errors > thresholds[p]))
        
        # Statistics - convert all numpy types to Python native types
        stats = {
            'test': {
                'mean_error': float(np.mean(test_errors)),
                'std_error': float(np.std(test_errors)),
                'sample_count': int(len(test_errors)),
                'min_error': float(np.min(test_errors)),
                'max_error': float(np.max(test_errors)),
                'median_error': float(np.median(test_errors))
            },
            'anomaly_thresholds': thresholds,
            'anomaly_counts': anomaly_counts
        }
        
        # Save test results
        results_path = self.artifacts_path / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✅ Test results saved: {results_path}")
        
        # Log to MLflow
        if mlflow_enabled and self.mlflow_tracker:
            # Convert metrics to Python native types
            metrics = {
                'test_mean_error': float(stats['test']['mean_error']),
                'test_std_error': float(stats['test']['std_error']),
                'test_samples': int(stats['test']['sample_count']),
                'test_min_error': float(stats['test']['min_error']),
                'test_max_error': float(stats['test']['max_error']),
                'test_median_error': float(stats['test']['median_error'])
            }
            
            for p, count in anomaly_counts.items():
                metrics[f'test_anomalies_{p}'] = int(count)
                metrics[f'test_anomaly_rate_{p}'] = float(count / len(test_errors))
                metrics[f'test_threshold_{p}'] = float(thresholds[p])
            
            try:
                self.mlflow_tracker.log_phase_metrics(metrics)
                
                # Log model to MLflow Model Registry
                print(f"  Logging model to MLflow...")
                
                input_dim = self.config.get('input_dim', len(self.params['data']['features']))
                input_example = np.random.randn(1, self.ticks_per_window, input_dim).astype(np.float32)
                
                # Prepare metadata for logging
                metadata = {
                    'window_size': int(self.ticks_per_window),
                    'input_dim': int(input_dim),
                    'symbol': self.symbol,
                    'test_samples': int(len(test_errors))
                }
                
                model_logged = self.mlflow_tracker.log_pytorch_model(
                    model=self.model,
                    model_name=f"{self.symbol}_Window_Autoencoder",
                    input_example_np=input_example,
                    metadata=metadata
                )
                
                if model_logged:
                    print(f"  ✅ Model logged to MLflow Model Registry")
                else:
                    print(f"  ⚠ Failed to log model to MLflow")
                
            except Exception as e:
                print(f"    ⚠ Warning: Could not log to MLflow: {e}")
            
            # End run
            try:
                self.mlflow_tracker.end_phase_run()
            except Exception as e:
                print(f"    ⚠ Warning: Could not end MLflow run: {e}")
        
        # Print results
        print(f"\n  Test Results:")
        print(f"    Mean Error: {stats['test']['mean_error']:.6f}")
        print(f"    Std Error: {stats['test']['std_error']:.6f}")
        print(f"    Samples: {stats['test']['sample_count']}")
        print(f"    Min Error: {stats['test']['min_error']:.6f}")
        print(f"    Max Error: {stats['test']['max_error']:.6f}")
        
        for p in percentiles:
            rate = anomaly_counts[p] / len(test_errors) * 100
            print(f"    Anomalies ({p}th %ile): {anomaly_counts[p]}/{len(test_errors)} ({rate:.1f}%)")
        
        # Save reconstruction errors for each window
        if hasattr(self.test_dataset, 'window_info'):
            window_errors = []
            for i, window_id in enumerate(self.test_dataset.window_info):
                if i < len(test_errors):
                    window_errors.append({
                        'window_id': str(window_id),
                        'reconstruction_error': float(test_errors[i]),
                        'is_anomaly_95': bool(test_errors[i] > thresholds.get(95, 0)),
                        'is_anomaly_99': bool(test_errors[i] > thresholds.get(99, 0))
                    })
            
            errors_path = self.artifacts_path / 'window_errors.json'
            with open(errors_path, 'w') as f:
                json.dump(window_errors, f, indent=2)
            print(f"  ✅ Window errors saved: {errors_path}")

if __name__ == "__main__":
    print("❌ This script should be run through FastAPI")
    print("   Use: python main.py")
    print("   Then access: http://127.0.0.1:9935/docs")