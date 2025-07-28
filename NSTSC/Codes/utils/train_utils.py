# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:30:37 2022

@author: yanru
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Models_node import TL_NN1, TL_NN2, TL_NN3, TL_NN4, TL_NN5, TL_NN6, init_weights

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.ln1 = torch.nn.LayerNorm(out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)
        self.ln2 = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(0.6)
        
        # Projection shortcut if dimensions don't match
        self.shortcut = None
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
            
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.ln2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        out = F.relu(out)
        return out

def create_classifier(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        ResidualBlock(input_dim, hidden_dim),
        ResidualBlock(hidden_dim, hidden_dim),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(hidden_dim, output_dim)
    )

def label_smoothing_loss(outputs, targets, smoothing=0.1):
    """
    Compute loss with label smoothing
    """
    n_classes = outputs.size(1)
    # Create smoothed labels
    targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
    smoothed_targets = targets_one_hot * (1 - smoothing) + smoothing / n_classes
    
    # Compute cross entropy with smoothed labels
    log_probs = F.log_softmax(outputs, dim=1)
    loss = -(smoothed_targets * log_probs).sum(dim=1).mean()
    return loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def Train_model(X_train_views, X_val_views, y_train, y_val, epochs=100, normalize_timeseries=True, device=None):
    """
    Train the NSTSC model for multi-class classification
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dimensions
    n_views = len(X_train_views)
    window_size = X_train_views[0].shape[1]
    n_classes = len(torch.unique(y_train))
    
    print(f"Number of views available: {n_views}")
    print(f"Window size: {window_size}")
    print(f"Number of classes: {n_classes}")
    
    # Initialize models
    models = {
        'TL_NN1': TL_NN1(window_size).to(device),
        'TL_NN2': TL_NN2(window_size).to(device),
        'TL_NN3': TL_NN3(window_size).to(device),
        'TL_NN4': TL_NN4(window_size).to(device),
        'TL_NN5': TL_NN5(window_size).to(device),
        'TL_NN6': TL_NN6(window_size).to(device)
    }
    
    # Add output layers for multi-class classification
    for name in models:
        models[name].fc = create_classifier(1, 128, n_classes).to(device)
        models[name].fc.apply(init_weights)
    
    # Initialize optimizers with reduced weight decay
    optimizers = {name: optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01, betas=(0.9, 0.999))
                 for name, model in models.items()}
    
    # Calculate warmup and total steps
    num_warmup_steps = epochs // 10  # 10% of total epochs for warmup
    num_training_steps = epochs
    
    # Initialize schedulers with warmup and cosine decay
    schedulers = {name: get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    ) for name, optimizer in optimizers.items()}
    
    # Training loop
    best_val_acc = 0
    best_model = None
    best_model_name = None
    patience = 30
    no_improve = 0
    best_epoch = 0
    gradient_accumulation_steps = 4  # Accumulate gradients for more stable training
    
    print("\nTraining configuration:")
    print(f"Initial learning rate: {optimizers[list(optimizers.keys())[0]].param_groups[0]['lr']}")
    print(f"Weight decay: {optimizers[list(optimizers.keys())[0]].param_groups[0]['weight_decay']}")
    print(f"Early stopping patience: {patience}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    for epoch in range(epochs):
        epoch_train_losses = []
        epoch_val_losses = []
        
        # Training phase
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            running_loss = 0.0
            
            # Process in smaller batches
            for i in range(0, len(X_train_views[0]), gradient_accumulation_steps):
                batch_end = min(i + gradient_accumulation_steps, len(X_train_views[0]))
                
                # Get batch data
                batch_views = [view[i:batch_end] for view in X_train_views]
                batch_labels = y_train[i:batch_end]
                
                # Forward pass
                intermediate_output = model(batch_views[0], batch_views[1], batch_views[2])
                outputs = model.fc(intermediate_output.unsqueeze(1))
                
                # Calculate loss with label smoothing
                loss = label_smoothing_loss(outputs, batch_labels)
                loss = loss / gradient_accumulation_steps  # Normalize loss
                running_loss += loss.item() * gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step optimizer after accumulating gradients
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(X_train_views[0]):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_train_losses.append(running_loss / len(X_train_views[0]))
            
            # Update learning rate
            schedulers[name].step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_intermediate = model(X_val_views[0], X_val_views[1], X_val_views[2])
                val_outputs = model.fc(val_intermediate.unsqueeze(1))
                val_loss = label_smoothing_loss(val_outputs, y_val)
                epoch_val_losses.append(val_loss.item())
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_model_name = name
                    best_epoch = epoch
                    no_improve = 0
        else:
                    no_improve += 1
        
        if (epoch + 1) % 5 == 0:
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Average training loss: {avg_train_loss:.4f}')
            print(f'  Average validation loss: {avg_val_loss:.4f}')
            print(f'  Best validation accuracy: {best_val_acc:.4f} (Model: {best_model_name}, Epoch: {best_epoch+1})')
            print(f'  Current learning rate: {optimizers[best_model_name].param_groups[0]["lr"]:.2e}')
        
        # Early stopping
        if no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    print(f'\nTraining completed:')
    print(f'Best model: {best_model_name}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best epoch: {best_epoch+1}')
    return best_model

def Evaluate_model(model, X_test_views, y_test, device=None):
    """
    Evaluate the trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        test_intermediate = model(X_test_views[0], X_test_views[1], X_test_views[2])
        test_outputs = model.fc(test_intermediate.unsqueeze(1))
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == y_test).float().mean().item()
    
    return test_acc

