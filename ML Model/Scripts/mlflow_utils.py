import mlflow
import mlflow.pytorch
import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class MLflowPhaseTracker:
    """Handles MLflow experiment tracking for different phases (train/val/test)"""
    
    def __init__(self, symbol: str, experiment_name_prefix: str = "TickAnomaly", 
                 tracking_uri: Optional[str] = None):
        """
        Args:
            symbol: Trading symbol
            experiment_name_prefix: Prefix for experiment names
            tracking_uri: MLflow tracking URI
        """
        self.symbol = symbol
        self.experiment_name = f"{experiment_name_prefix}_{symbol}"
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        self.setup_experiment()
        
        # Store run IDs for linking
        self.run_ids = {}
        self.active_run = None
    
    def setup_experiment(self):
        """Set up MLflow experiment for a symbol"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                print(f"    Created MLflow experiment: {self.experiment_name}")
            else:
                print(f"    Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"    ⚠ Warning: Could not setup MLflow experiment: {e}")
    
    def start_phase_run(self, phase: str, parent_run_id: Optional[str] = None, 
                       tags: Optional[Dict] = None) -> str:
        """Start a new MLflow run for a specific phase"""
        try:
            # Check for active run and end it if exists
            if mlflow.active_run():
                mlflow.end_run()
                
            run_name = f"{self.symbol}_{phase}_{datetime.now().strftime('%H%M%S')}"
            
            # Set tags
            default_tags = {
                "symbol": self.symbol,
                "phase": phase,
                "model_type": "Autoencoder",
                "framework": "PyTorch",
                "timestamp": datetime.now().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            if parent_run_id:
                default_tags["mlflow.parentRunId"] = parent_run_id
            
            # Start run
            self.active_run = mlflow.start_run(run_name=run_name, tags=default_tags)
            run_id = self.active_run.info.run_id
            self.run_ids[phase] = run_id
            
            print(f"    Started {phase} run: {run_name} (ID: {run_id[:8]}...)")
            return run_id
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not start {phase} run: {e}")
            return None
    
    def end_phase_run(self):
        """End the current active run"""
        if self.active_run:
            try:
                mlflow.end_run()
                self.active_run = None
            except Exception as e:
                print(f"    ⚠ Warning: Could not end run: {e}")
    
    def log_phase_params(self, params: Dict):
        """Log parameters for the current phase"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"    ⚠ Warning: Could not log parameters: {e}")
    
    def log_phase_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics for the current phase"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"    ⚠ Warning: Could not log metrics: {e}")
    
    def log_phase_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact for the current phase"""
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            print(f"    ⚠ Warning: Could not log artifact: {e}")
    
    def log_pytorch_model(self, model: nn.Module, model_name: str, 
                         input_example_np: np.ndarray, metadata: Optional[Dict] = None):
        """Log PyTorch model to MLflow correctly"""
        try:
            # Convert to dictionary format if needed
            if isinstance(input_example_np, np.ndarray):
                # Create a dictionary with a meaningful key
                input_example_dict = {"input": input_example_np}
            else:
                input_example_dict = input_example_np
            
            # Log the model with artifact_path parameter
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                input_example=input_example_dict,
                registered_model_name=model_name
            )
            
            # Log metadata as tags if provided
            if metadata:
                mlflow.set_tags(metadata)
            
            print(f"    ✓ Model logged to MLflow Model Registry: {model_name}")
            return True
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not log model to MLflow: {e}")
            
            # Fallback: try without registered_model_name
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    input_example=input_example_dict
                )
                print(f"    ✓ Model logged to current run (fallback)")
                return True
            except Exception as e2:
                print(f"    ⚠ Could not log model: {e2}")
                
                # Final fallback: save as artifact
                try:
                    model_path = f"temp_{model_name}.pth"
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, artifact_path="models")
                    os.remove(model_path)
                    print(f"    ✓ Model saved as artifact")
                    return True
                except Exception as e3:
                    print(f"    ⚠ Could not save model as artifact: {e3}")
                    return False