import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, time
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import gc
import sys
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

class WindowTickDataDataset(Dataset):
    """PyTorch Dataset for window-based tick data - 600 ticks per window"""
    def __init__(self, data_dict, features, ticks_per_window=600):
        """
        Args:
            data_dict: Dictionary with window-based data
            features: List of feature column names
            ticks_per_window: Target ticks per window (default 600)
        """
        self.features = features
        self.ticks_per_window = ticks_per_window
        
        # Vectorized extraction of all samples
        all_samples = []
        self.window_info = []  # Store (date, window_type) for each sample
        
        for window_id, df in data_dict.items():
            X_window = df[features].values.astype(np.float32)
            
            # Ensure window has correct number of ticks
            if len(X_window) != ticks_per_window:
                X_window = self._pad_window(X_window)
            
            all_samples.append(X_window)
            self.window_info.append(window_id)
        
        # Stack all windows (n_windows, ticks_per_window, n_features)
        self.X = np.stack(all_samples)
        self.X_tensor = torch.from_numpy(self.X).float()
        
    def _pad_window(self, window_data):
        """Pad or truncate window to target ticks"""
        n_ticks = len(window_data)
        
        if n_ticks == self.ticks_per_window:
            return window_data
        
        # Create padded array
        padded = np.zeros((self.ticks_per_window, len(self.features)), dtype=np.float32)
        
        if n_ticks < self.ticks_per_window:
            # Place existing ticks at evenly spaced positions
            indices = np.linspace(0, self.ticks_per_window - 1, n_ticks, dtype=int)
            padded[indices] = window_data
            
            # Forward fill
            for i in range(1, self.ticks_per_window):
                if np.all(padded[i] == 0):
                    padded[i] = padded[i-1]
            
            # If first value is zero, backward fill
            if np.all(padded[0] == 0) and self.ticks_per_window > 1:
                for i in range(1, self.ticks_per_window):
                    if not np.all(padded[i] == 0):
                        padded[:i] = padded[i]
                        break
        else:
            # Take evenly spaced samples
            indices = np.linspace(0, n_ticks - 1, self.ticks_per_window, dtype=int)
            padded = window_data[indices]
        
        return padded
    
    def __len__(self):
        return len(self.X_tensor)
    
    def __getitem__(self, idx):
        return self.X_tensor[idx]
    
    def get_window_info(self, idx):
        """Get window information for a given index"""
        return self.window_info[idx]

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection in window-based tick data"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=16, 
                 ticks_per_window=600, dropout_rate=0.1, activation='relu', 
                 use_batch_norm=False, weight_decay=0.001):
        """
        Args:
            input_dim: Features per tick (Bid, Ask, Spread = 3)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            ticks_per_window: Number of ticks per window (default 600)
            dropout_rate: Dropout probability
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            weight_decay: L2 regularization strength
        """
        super(Autoencoder, self).__init__()
        self.ticks_per_window = ticks_per_window
        self.total_input_dim = input_dim * ticks_per_window
        self.weight_decay = weight_decay
        
        # Get activation function
        self.activation = self._get_activation(activation)
        
        # Encoder layers
        encoder_layers = []
        prev_dim = self.total_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, self.total_input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_activation(self, activation_name):
        """Get activation function by name"""
        activation_name = activation_name.lower()
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'elu':
            return nn.ELU(alpha=1.0)
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
    
    def forward(self, x):
        # x shape: (batch_size, ticks_per_window, input_dim)
        batch_size = x.shape[0]
        
        # Flatten: (batch_size, ticks_per_window * input_dim)
        x_flat = x.reshape(batch_size, -1)
        
        encoded = self.encoder(x_flat)
        decoded_flat = self.decoder(encoded)
        
        # Reshape back: (batch_size, ticks_per_window, input_dim)
        decoded = decoded_flat.reshape(batch_size, self.ticks_per_window, -1)
        
        return decoded, encoded
    
    def get_reconstruction_error(self, x):
        """Compute reconstruction error (MSE) for anomaly detection"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error

def filter_time_windows(df):
    """Filter data for specific time windows: 7:50-8:00 and 13:50-14:00"""
    if df.empty:
        return df
    
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Time'] = df['DateTime'].dt.time
    
    morning_start = time(7, 50)
    morning_end = time(8, 0)
    afternoon_start = time(13, 50)
    afternoon_end = time(14, 0)
    
    mask = (
        ((df['Time'] >= morning_start) & (df['Time'] <= morning_end)) |
        ((df['Time'] >= afternoon_start) & (df['Time'] <= afternoon_end))
    )
    
    return df[mask].copy()

def compute_spread(df):
    """Vectorized computation of Spread column from Bid and Ask"""
    if 'Bid' not in df.columns or 'Ask' not in df.columns:
        return df
    
    df = df.copy()
    df['Spread'] = np.maximum(df['Ask'] - df['Bid'], 0)
    
    return df

def prepare_tick_features(df):
    """Prepare tick features with vectorized operations"""
    df = compute_spread(df)
    
    required_features = ['Bid', 'Ask', 'Spread']
    available_features = []
    
    for feature in required_features:
        if feature in df.columns:
            # Vectorized cleaning
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].ffill().bfill().fillna(0)
            available_features.append(feature)
    
    return df, available_features

def create_window_based_data(df, features, ticks_per_window=600):
    """
    Create window-based data from filtered data
    Returns dict: {window_id: window_dataframe}
    """
    df = df.copy()
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    
    # Define window boundaries
    morning_start = time(7, 50)
    morning_end = time(8, 0)
    afternoon_start = time(13, 50)
    afternoon_end = time(14, 0)
    
    window_data = {}
    
    # Process each date separately
    for date, date_group in df.groupby('Date'):
        # Morning window
        morning_data = date_group[
            (date_group['Time'] >= morning_start) & 
            (date_group['Time'] <= morning_end)
        ].copy()
        
        if not morning_data.empty:
            morning_data = morning_data.sort_values('DateTime')
            morning_id = f"{date}_morning"
            window_data[morning_id] = morning_data[features].reset_index(drop=True)
        
        # Afternoon window
        afternoon_data = date_group[
            (date_group['Time'] >= afternoon_start) & 
            (date_group['Time'] <= afternoon_end)
        ].copy()
        
        if not afternoon_data.empty:
            afternoon_data = afternoon_data.sort_values('DateTime')
            afternoon_id = f"{date}_afternoon"
            window_data[afternoon_id] = afternoon_data[features].reset_index(drop=True)
    
    return window_data

def prepare_window_data(df, features, ticks_per_window=600):
    """
    Prepare window-based data with padding
    Returns dict: {window_id: padded_dataframe}
    """
    # Get window-based data
    window_data_raw = create_window_based_data(df, features, ticks_per_window)
    
    # Pad each window to target ticks
    window_data_padded = {}
    
    for window_id, window_df in window_data_raw.items():
        n_ticks = len(window_df)
        
        if n_ticks == 0:
            continue
        
        # Create padded array
        padded = np.zeros((ticks_per_window, len(features)), dtype=np.float32)
        
        if n_ticks < ticks_per_window:
            # Place existing ticks at evenly spaced positions
            indices = np.linspace(0, ticks_per_window - 1, n_ticks, dtype=int)
            padded[indices] = window_df.values
            
            # Forward fill
            for i in range(1, ticks_per_window):
                if np.all(padded[i] == 0):
                    padded[i] = padded[i-1]
            
            # If first value is zero, backward fill
            if np.all(padded[0] == 0) and ticks_per_window > 1:
                for i in range(1, ticks_per_window):
                    if not np.all(padded[i] == 0):
                        padded[:i] = padded[i]
                        break
        else:
            # Take evenly spaced samples
            indices = np.linspace(0, n_ticks - 1, ticks_per_window, dtype=int)
            padded = window_df.values[indices]
        
        # Create dataframe
        padded_df = pd.DataFrame(padded, columns=features)
        padded_df['Window_ID'] = window_id
        padded_df['Original_Tick_Count'] = n_ticks
        
        window_data_padded[window_id] = padded_df
    
    return window_data_padded

def split_window_data(window_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split window data into train, validation, and test sets"""
    window_ids = sorted(window_data.keys())
    n_windows = len(window_ids)
    
    train_idx = int(n_windows * train_ratio)
    val_idx = train_idx + int(n_windows * val_ratio)
    
    train_windows = window_ids[:train_idx]
    val_windows = window_ids[train_idx:val_idx]
    test_windows = window_ids[val_idx:]
    
    train_data = {wid: window_data[wid] for wid in train_windows}
    val_data = {wid: window_data[wid] for wid in val_windows}
    test_data = {wid: window_data[wid] for wid in test_windows}
    
    window_info = {
        'train_windows': train_windows,
        'val_windows': val_windows,
        'test_windows': test_windows,
        'total_windows': n_windows
    }
    
    return train_data, val_data, test_data, window_info

def scale_window_datasets(train_data, val_data, test_data, features):
    """Scale all window datasets using the same scaler fitted on training data"""
    # Extract all training data for scaler fitting
    all_train_samples = []
    for window_df in train_data.values():
        all_train_samples.append(window_df[features].values)
    
    X_train = np.vstack(all_train_samples)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Scale all datasets
    def scale_single_dataset(data_dict):
        scaled_dict = {}
        for window_id, df in data_dict.items():
            X = df[features].values
            X_scaled = scaler.transform(X)
            
            scaled_df = pd.DataFrame(X_scaled, columns=features)
            scaled_df['Window_ID'] = window_id
            if 'Original_Tick_Count' in df.columns:
                scaled_df['Original_Tick_Count'] = df['Original_Tick_Count']
            
            scaled_dict[window_id] = scaled_df
        return scaled_dict
    
    train_data_scaled = scale_single_dataset(train_data)
    val_data_scaled = scale_single_dataset(val_data)
    test_data_scaled = scale_single_dataset(test_data)
    
    return train_data_scaled, val_data_scaled, test_data_scaled, scaler

def compute_reconstruction_errors_vectorized(model, data_loader, device):
    """Vectorized computation of reconstruction errors"""
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            errors = model.get_reconstruction_error(batch)
            all_errors.append(errors.cpu().numpy())
    
    if all_errors:
        return np.concatenate(all_errors)
    return np.array([])

def load_parameters(params_path):
    """Load model parameters from JSON file"""
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params