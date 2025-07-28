# -*- coding: utf-8 -*-

from Models_node import *
from utils.dataloader import load_eeg_data, prepare_multiview_data, adjust_labels
from utils.train_utils import *
import torch

def main():
    # CUDA Configuration
    if torch.cuda.is_available():
        device = torch.device('cuda:2')  # Explicitly use GPU 2
        print(f'Using CUDA device: {torch.cuda.get_device_name(2)}')
        print(f'Device capabilities: {torch.cuda.get_device_capability(2)}')
        print(f'Current device: {torch.cuda.current_device()}')
        torch.cuda.set_device(2)  # Set default device
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Using CPU')
    
    # Configuration
    data_path = "/home/belyaeva.a/df_open_fixed.csv"
    window_size = 10  # Reduced window size to 10 points
    normalize_dataset = True
    max_epochs = 200  # Increased max epochs
    
    print('Loading and preprocessing EEG data...')
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_eeg_data(
        data_path, 
        test_size=0.2,
        val_size=0.2,
        device=device
    )
    
    print("\nData shapes after loading:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Prepare multi-view data with overlapping windows
    X_train_views, X_val_views, X_test_views, n_train_windows, n_val_windows, n_test_windows = prepare_multiview_data(
        X_train, X_val, X_test,
        window_size=window_size,
        device=device
    )
    
    print("\nData shapes after view preparation:")
    print(f"Number of windows per signal: {n_train_windows}")
    for i, view in enumerate(X_train_views):
        print(f"Train view {i} shape: {view.shape}")
    for i, view in enumerate(X_val_views):
        print(f"Val view {i} shape: {view.shape}")
    for i, view in enumerate(X_test_views):
        print(f"Test view {i} shape: {view.shape}")
    
    # Adjust labels for windowed data
    y_train_adj = adjust_labels(y_train, n_train_windows).to(device)
    y_val_adj = adjust_labels(y_val, n_val_windows).to(device)
    y_test_adj = adjust_labels(y_test, n_test_windows).to(device)
    
    print("\nLabel shapes:")
    print(f"y_train shape: {y_train_adj.shape}")
    print(f"y_val shape: {y_val_adj.shape}")
    print(f"y_test shape: {y_test_adj.shape}")
    
    # Get dataset dimensions
    N = X_train_views[0].shape[0]  # Number of samples after windowing
    T = window_size  # Time steps per sample
    
    print(f'\nDataset dimensions after preprocessing:')
    print(f'Training samples: {N}')
    print(f'Time steps per sample: {T}')
    print(f'Number of views: {len(X_train_views)}')
    
    # Train model
    print('\nTraining model...')
    Tree = Train_model(
        X_train_views, 
        X_val_views,
        y_train_adj,
        y_val_adj,
        epochs=max_epochs,
        normalize_timeseries=normalize_dataset,
        device=device
    )
    
    # Evaluate model
    test_accuracy = Evaluate_model(Tree, X_test_views, y_test_adj, device=device)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()



