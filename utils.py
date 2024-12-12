
import torch
import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import openml
import pandas as pd
import wandb
import matplotlib.pyplot as plt

def log_weight_distributions(model, epoch, model_name="HashedNetwork"):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            # Log weight distribution as a histogram
            wandb.log({f"{model_name}/{name}_weights_epoch_{epoch}": wandb.Histogram(param.data.cpu().numpy())})

def get_higgs_small_dataset():
    dataset = openml.datasets.get_dataset(23512)  # Higgs Small OpenML ID
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    X = pd.DataFrame(X)
   
    X = X.select_dtypes(include=[np.number])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.cat.codes.to_numpy()

    
    return X, y

def plot_layerwise_weight_distribution(model, model_name):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(weights, bins=100, alpha=0.75, label=f"{name} Weights")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.title(f"Weight Distribution for {model_name} - Layer: {name}")
            plt.legend()
            plt.grid()
            plt.savefig(f"weight_distribution{model_name}_{name}.png")  # Save the plot as an image

            plt.show()


def plot_results(hashed_loss, vanilla_loss):
    epochs = list(range(1, len(hashed_loss) + 1))
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hashed_loss, label="Hashed Network Loss", linestyle='-', marker='o')
    plt.plot(epochs, vanilla_loss, label="Vanilla Network Loss", linestyle='--', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Between Hashed and Vanilla Networks")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_comparison.png")  # Save the plot as an image

    plt.show()


def generate_synthetic_data(n_samples=10000, n_features=30, n_classes=10):
    # Generate base dataset with feature complexity
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(0.6 * n_features),
        n_redundant=int(0.2 * n_features),
        n_repeated=int(0.1 * n_features),
        n_classes=n_classes,
        weights=[0.7] + [0.3 / (n_classes - 1)] * (n_classes - 1),  # Class imbalance
        flip_y=0.05,  # Add noise to labels
        random_state=42
    )

    # Add non-linear transformations
    X[:, 0] = np.sin(X[:, 0])
    X[:, 1] = X[:, 1] ** 2

    # Add mixed distributions
    def generate_mixed_features(X):
        X_uniform = np.random.uniform(-1, 1, size=(X.shape[0], X.shape[1] // 3))
        X_categorical = np.random.choice([0, 1], size=(X.shape[0], X.shape[1] // 3))
        return np.hstack([X, X_uniform, X_categorical])

    X = generate_mixed_features(X)

    # Add noise
    noise = np.random.normal(0, 0.1, X.shape)
    X += noise

    # Add lagged features (simulate time-series)
    for lag in range(1, 3):  # Add 2 lagged features
        X = np.hstack([X, np.roll(X, lag, axis=0)])

    # Preprocessing: scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
