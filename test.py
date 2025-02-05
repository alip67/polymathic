import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import HDF5Dataset
from CNN_classifier import BasicCNNClassifier
from model import VisionTransformer, TinyViT
import matplotlib.pyplot as plt
import numpy as np

import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, scheduler_type="StepLR", early_stopping_patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function (for binary classification, use BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()  # Use CrossEntropyLoss() if multi-class
    
    # Optimizer (Adam works well with ViTs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler
    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None  # No scheduler
    
    # Track metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        ### ========== TRAINING ==========
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device).float().squeeze()  # Ensure binary labels
            
            optimizer.zero_grad()
            # outputs = model(inputs)
            outputs = model(inputs[:,0,:,:].unsqueeze(dim=1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss
            running_train_loss += loss.item() * inputs.size(0)

            # Track accuracy
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_train_loss / total_train
        train_accuracy = correct_train / total_train

        ### ========== VALIDATION ==========
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device).float().squeeze()

                outputs = model(inputs[:,0,:,:].unsqueeze(dim=1))
                # outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_val_loss / total_val
        val_accuracy = correct_val / total_val

        # Track losses and accuracy
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Learning rate scheduler step
        if scheduler:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)  # Adjust based on validation loss
            else:
                scheduler.step()

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, "best_model.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break
    

    # Plot results
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path="training_plot.png"):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")

        # Save and Show the plot
    plt.savefig(save_path, dpi=300)
    print(f"Training plot saved to {save_path}")
    plt.show()
 
# Set seed for reproducibility
def set_seed(seed=42):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_test_predictions(model, data_loader, device):
    """ Writes test predictions to a outputs/predictions.txt 
        with one prediction per line as in the example file"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs[:, 0, :, :].unsqueeze(dim=1))
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            predictions.extend(preds.flatten())
    
    with open("outputs/predictions.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def load_best_model(model, device, model_path="best_model.pth"):
    """Loads the best saved model state."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension of the model')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--input_shape', type=int, default=28, help='Input shape size')
    parser.add_argument('--proj_dim', type=int, default=32, help='Projection dimension')
    parser.add_argument('--mlp_dim', type=int, default=128, help='MLP hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 regularization)')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for vision transformer')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for transformer')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TinyViT(
        input_dim=args.input_dim,
        input_shape=args.input_shape,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    )
    
    train_dataset = HDF5Dataset("data/train.hdf5")   
    print("Class Distribution:", torch.bincount(torch.tensor(train_dataset.labels, dtype=torch.int).squeeze()))
    
    val_dataset = HDF5Dataset("data/valid.hdf5")
    print("Class Distribution:", torch.bincount(torch.tensor(val_dataset.labels, dtype=torch.int).squeeze()))

    test_dataset = HDF5Dataset("data/test.hdf5")
    print("Class Distribution:", torch.bincount(torch.tensor(val_dataset.labels, dtype=torch.int).squeeze()))
    

    # train_recording_date = (train_dataset.inputs[:, 1, :, :] - 0) / (6 - 0)
    # train_density_modified = np.add(train_dataset.inputs[:, 0, :, :], train_recording_date)
    # train_dataset.inputs = np.stack([train_density_modified, train_recording_date], axis=1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    train_model(model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.learning_rate, scheduler_type="StepLR", early_stopping_patience=args.early_stopping_patience)
    model = load_best_model(model, device)
    write_test_predictions(model, test_loader, device)

