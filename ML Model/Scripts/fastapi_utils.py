import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def load_parameters(json_path: str) -> Dict:
    """Load all parameters from JSON file"""
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Parameters file not found: {json_path}")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        raise

def get_paths_from_params(params: Dict) -> Dict[str, Path]:
    """Extract and validate paths from parameters"""
    paths_section = params.get('paths', {})
    
    paths = {
        'artifacts': Path(paths_section.get('artifacts_path', '')),
        'data': Path(paths_section.get('data_path', '')),
        'symbols': Path(paths_section.get('symbols_path', '')),
        'scripts': Path(paths_section.get('scripts_path', ''))
    }
    
    # Validate required paths
    required = ['artifacts', 'data', 'symbols']
    for key in required:
        if not paths[key].exists():
            raise FileNotFoundError(f"Path not found: {paths[key]}")
    
    return paths

def validate_parameters(params: Dict) -> bool:
    """Validate that all required parameters are present"""
    required_sections = ['paths', 'mlflow', 'data', 'model', 'training']
    
    for section in required_sections:
        if section not in params:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate data ratios sum to 1
    data = params['data']
    ratios = [data.get('train_ratio', 0), data.get('val_ratio', 0), data.get('test_ratio', 0)]
    if abs(sum(ratios) - 1.0) > 0.01:  # Allow small floating point error
        raise ValueError(f"Data ratios don't sum to 1. Got: {ratios}")
    
    return True

def get_trained_symbols(artifacts_path: Path) -> list:
    """Get list of symbols that have been trained"""
    trained_symbols = []
    if artifacts_path.exists():
        for item in artifacts_path.iterdir():
            if item.is_dir():
                model_path = item / 'trained_autoencoder.pth'
                best_model_path = item / 'best_model.pth'
                test_data_path = item / 'test_data.pkl'
                
                if (model_path.exists() or best_model_path.exists()) and test_data_path.exists():
                    trained_symbols.append(item.name)
    return sorted(trained_symbols)

def create_response(success: bool, message: str, data: Dict = None) -> Dict:
    """Create standardized API response"""
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    if data:
        response["data"] = data
    return response