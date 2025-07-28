import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_eeg_data(file_path, test_size=0.2, val_size=0.2, random_state=42, device=None):
    """
    Load and preprocess EEG data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        device (torch.device): Device to move tensors to
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Read data
    print("Loading data from CSV...")
    df = pd.read_csv(file_path)
    
    # Get EEG channel columns (excluding problematic channels A1, A2, Fpz)
    eeg_channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2', 'Fz', 'Cz', 'Pz']
    
    print(f"Using {len(eeg_channels)} EEG channels: {eeg_channels}")
    
    # First, determine the length of time series
    sample_lengths = []
    for _, row in df.iterrows():
        row_lengths = set()
        for channel in eeg_channels:
            try:
                if pd.isna(row[channel]):
                    continue
                channel_data = eval(str(row[channel]))
                if isinstance(channel_data, list):
                    row_lengths.add(len(channel_data))
            except:
                continue
        # Only add length if all channels in the row have the same length
        if len(row_lengths) == 1:  # All channels have same length
            sample_lengths.append(row_lengths.pop())
    
    if not sample_lengths:
        raise ValueError("Could not determine time series length from data")
    
    # Use the most common length
    target_length = max(set(sample_lengths), key=sample_lengths.count)
    print(f"Using time series length of {target_length}")
    
    # Extract features and convert string representations to arrays
    X_data = []
    valid_indices = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        sample = []
        valid = True
        row_lengths = set()
        
        # First pass: check all channel lengths
        for channel in eeg_channels:
            try:
                if pd.isna(row[channel]):
                    valid = False
                    break
                    
                channel_data = eval(str(row[channel]))
                if isinstance(channel_data, list):
                    row_lengths.add(len(channel_data))
                else:
                    valid = False
                    break
            except:
                valid = False
                break
        
        # Skip if channels have different lengths
        if len(row_lengths) != 1:
            valid = False
            skipped_count += 1
            continue
            
        # Second pass: process data if all lengths are valid
        if valid:
            for channel in eeg_channels:
                try:
                    channel_data = eval(str(row[channel]))
                    # Pad or truncate to target length
                    if len(channel_data) > target_length:
                        channel_data = channel_data[:target_length]
                    elif len(channel_data) < target_length:
                        channel_data = channel_data + [0] * (target_length - len(channel_data))
                    sample.append(channel_data)
                except:
                    valid = False
                    break
        
        if valid and len(sample) == len(eeg_channels):
            X_data.append(np.array(sample))
            valid_indices.append(idx)
    
    print(f"Skipped {skipped_count} samples due to inconsistent channel lengths")
    
    if not X_data:
        raise ValueError("No valid data found after processing")
    
    X = np.array(X_data)
    print(f"Processed data shape: {X.shape}")
    
    # Get labels
    y = df.iloc[valid_indices]['label'].values
    print(f"Original labels: {np.unique(y)}")
    
    # Convert labels to numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Split into train+val and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train+val into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size/(1-test_size),  # Adjust val_size proportion
        random_state=random_state,
        stratify=y_temp
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Move to device if specified
    if device is not None:
        X_train = X_train.to(device)
        X_val = X_val.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_val = y_val.to(device)
        y_test = y_test.to(device)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_multiview_data(X_train, X_val, X_test, window_size=50, device=None):
    """
    Prepare multi-view data by creating sliding windows
    
    Args:
        X_train (torch.Tensor): Training data
        X_val (torch.Tensor): Validation data
        X_test (torch.Tensor): Test data
        window_size (int): Size of sliding window
        device (torch.device): Device to move tensors to
        
    Returns:
        tuple: (X_train_views, X_val_views, X_test_views)
    """
    def create_views(X):
        # X shape: (batch_size, n_channels, time_series_length)
        n_samples = X.shape[0]
        n_channels = X.shape[1]
        time_length = X.shape[2]
        
        # Split channels into groups based on brain regions
        frontal_idx = [0, 1, 8, 9, 14, 15, 16]  # F3, F4, F7, F8, Fp1, Fp2, Fz
        central_idx = [2, 3, 17, 10, 11]        # C3, C4, Cz, T3, T4
        posterior_idx = [4, 5, 18, 6, 7, 12, 13] # P3, P4, Pz, O1, O2, T5, T6
        
        # Create views by averaging channels in each region
        frontal_view = X[:, frontal_idx, :].mean(dim=1, keepdim=True)   # (batch_size, 1, time_series_length)
        central_view = X[:, central_idx, :].mean(dim=1, keepdim=True)   # (batch_size, 1, time_series_length)
        posterior_view = X[:, posterior_idx, :].mean(dim=1, keepdim=True) # (batch_size, 1, time_series_length)
        
        # Use windows with 75% overlap
        stride = window_size * 3 // 4  # 75% overlap
        
        # Calculate number of windows
        n_windows = (time_length - window_size) // stride + 1
        
        # Initialize tensors to store windows
        frontal_windows = torch.zeros((n_samples, n_windows, window_size), device=X.device)
        central_windows = torch.zeros((n_samples, n_windows, window_size), device=X.device)
        posterior_windows = torch.zeros((n_samples, n_windows, window_size), device=X.device)
        
        # Extract windows
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            frontal_windows[:, i, :] = frontal_view.squeeze(1)[:, start_idx:end_idx]
            central_windows[:, i, :] = central_view.squeeze(1)[:, start_idx:end_idx]
            posterior_windows[:, i, :] = posterior_view.squeeze(1)[:, start_idx:end_idx]
        
        # Normalize each window
        frontal_windows = (frontal_windows - frontal_windows.mean(dim=2, keepdim=True)) / (frontal_windows.std(dim=2, keepdim=True) + 1e-6)
        central_windows = (central_windows - central_windows.mean(dim=2, keepdim=True)) / (central_windows.std(dim=2, keepdim=True) + 1e-6)
        posterior_windows = (posterior_windows - posterior_windows.mean(dim=2, keepdim=True)) / (posterior_windows.std(dim=2, keepdim=True) + 1e-6)
        
        # Reshape to (batch_size * n_windows, window_size)
        frontal_final = frontal_windows.reshape(-1, window_size)
        central_final = central_windows.reshape(-1, window_size)
        posterior_final = posterior_windows.reshape(-1, window_size)
        
        return [frontal_final, central_final, posterior_final], n_windows
    
    X_train_views, n_train_windows = create_views(X_train)
    X_val_views, n_val_windows = create_views(X_val)
    X_test_views, n_test_windows = create_views(X_test)
    
    # Move to device if specified
    if device is not None:
        X_train_views = [view.to(device) for view in X_train_views]
        X_val_views = [view.to(device) for view in X_val_views]
        X_test_views = [view.to(device) for view in X_test_views]
    
    return X_train_views, X_val_views, X_test_views, n_train_windows, n_val_windows, n_test_windows

def adjust_labels(y, n_windows):
    """
    Adjust labels to match the windowed data structure
    Args:
        y (torch.Tensor): Labels tensor
        n_windows (int): Number of windows per sample
    Returns:
        torch.Tensor: Repeated labels for each window
    """
    # Repeat each label for each window of the corresponding signal
    return y.repeat_interleave(n_windows) 