from lsh import LSH
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from simHash import SimHash
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class HashedFC(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super(HashedFC, self).__init__()
        self.params = nn.Linear(input_dim, output_dim)
        self.params.bias = nn.Parameter(torch.Tensor(output_dim))  # 1D bias
        self.K = 5  # Amount of Hash Functions
        self.D = input_dim  # Input dimension
        self.L = 1  # Single hash table
        self.hash_weight = None  # Optionally provide custom hash weights
        self.num_class = output_dim  # Number of output classes
        #self.init_weights(self.params.weight, self.params.bias)

        # Activation counters and running metrics
        self.running_activations = torch.zeros(output_dim).to(device)
        self.lsh = None
        self.initializeLSH()

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

    def initializeLSH(self):
        simhash = SimHash(self.D + 1, self.K, self.L, self.hash_weight)
        self.lsh = LSH(simhash, self.K, self.L)
        weight_tolsh = torch.cat((self.params.weight, self.params.bias.unsqueeze(1)), dim=1)
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class)

    def rebuildLSH(self):
        self.lsh.clear()
        self.lsh.setSimHash(SimHash(self.D + 1, self.K, self.L))
        weight_tolsh = torch.cat((self.params.weight, self.params.bias.unsqueeze(1)), dim=1)
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class )



    def accumulate_metrics(self, activations):
        for idx, activation in enumerate(activations):
            self.running_activations[idx] += activation.item()

    def select_representatives(self, bucket_indices):
        """
        Select representatives based on the weighted average of activations and weights.
        """

        representatives = []
        for bucket in bucket_indices:
            if len(bucket) == 0:
                continue

            # Initialize sums for weighted average computation
            weighted_sum = 0
            count = 0
            rep_idx = 0

            # Compute weighted sum of weights and count total activations
            for idx in bucket:
                activation = self.running_activations[idx]
                weight = self.params.weight[idx]
                weighted_sum += activation * weight
                count += 1

                if self.running_activations[idx] > self.running_activations[rep_idx]:
                    rep_idx = idx

            # Calculate representative weight
            if count > 0:
                representative_weight = weighted_sum / count
            else:
                representative_weight = torch.mean(self.params.weight[bucket], dim=0)

            # Find the index of the weight closest to the representative weight
           
            representatives.append(rep_idx)
        

        return representatives  # Return as a list of representative indices


    def prune_weights(self, representatives, input_dim):
        """
        Prune weights based on the selected representatives.
        Adjust the dimensions of the layer accordingly.
        """
        # Ensure `representatives` is a tensor of indices
        keep_indices = torch.tensor(representatives, dtype=torch.long, device=device)

        # Slice rows (output dimensions) and keep all columns for now
        pruned_weights = self.params.weight[keep_indices, :input_dim]  # Slice rows and columns
        pruned_biases = self.params.bias[keep_indices]                 # Slice only rows for biases

        # Check the shapes of pruned weights and biases
        #print(f"Pruned weights shape: {pruned_weights.shape}, Pruned biases shape: {pruned_biases.shape}")

        # Update the number of classes (output dimensions)
        self.num_class = len(representatives)
        self.D = input_dim  # Number of input features

        # Verify dimensions
        #print(f"Input dim: {self.D}, Reps: {self.num_class}")

        # Initialize a new Linear layer with pruned dimensions
        new_layer = nn.Linear(self.D, self.num_class).to(device)

        # Copy pruned weights and biases into the new layer
        new_layer.weight.data.copy_(pruned_weights)
        new_layer.bias.data.copy_(pruned_biases)
        print(f"New layer after assignment: {new_layer.weight.shape}, {new_layer.bias.shape}")

        # Replace the old layer
        self.params = new_layer

        # Rebuild LSH
        self.rebuildLSH()
        #print(f"Pruned weights. Kept indices: {representatives}")




    def update_weights(self, input_dim):
        bucket_indices = self.lsh.representatives()
        representatives = self.select_representatives(bucket_indices)
        self.prune_weights(representatives, input_dim)

    def forward(self, x):
        return self.params(x)


class HashedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HashedNetwork, self).__init__()
        self.fc1 = HashedFC(input_dim, hidden_dim, K=20)
        self.fc2 = HashedFC(hidden_dim, hidden_dim, K=15)
        self.fc3 = HashedFC(hidden_dim, output_dim, K=5)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        activations = torch.sum(x, dim=0).detach()
        self.fc1.accumulate_metrics(activations)
        x = F.relu(self.fc2(x))
        self.fc2.accumulate_metrics(torch.sum(x, dim=0).detach())
        x = self.fc3(x)
        self.fc3.accumulate_metrics(torch.sum(x, dim=0).detach())
        return x


class VanillaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Generate synthetic data
def generate_synthetic_data(n_samples=10000, n_features=30, n_classes=10):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Train function
def train_model(model, optimizer, criterion, X_train, y_train, epochs=20, prune_every=19):
    """
    Train the model and prune hashed layers every 'prune_every' epochs.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Dynamically prune and adjust hashed layers every 'prune_every' epochs
        if isinstance(model, HashedNetwork) and (epoch + 1) % prune_every == 5:
            print(f"\nEpoch {epoch + 1}: Pruning weights and adjusting dimensions...")
            
            # Prune fc1 and propagate changes to fc2
            model.fc1.update_weights(X_train.shape[1])

            # Prune fc2 and propagate changes to fc3
            model.fc2.update_weights(model.fc1.num_class)

            model.fc3.prune_weights(torch.arange(model.fc3.num_class, device=device), model.fc2.num_class)



# Function to measure accuracy
def measure_accuracy(model, X, y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        correct = (predicted == y).sum().item()  # Count correct predictions
        total = y.size(0)  # Total number of samples
    accuracy = correct / total * 100
    return accuracy


# Main comparison
def compare_networks():
    input_dim = 30
    hidden_dim = 10000
    output_dim = 2

    # Generate data
    X, y = generate_synthetic_data(n_samples=10000, n_features=input_dim, n_classes=output_dim)
    X_train, y_train = X.to(device), y.to(device)

    # Hashed Network
    hashed_model = HashedNetwork(input_dim, hidden_dim, output_dim).to(device)
    hashed_optimizer = torch.optim.Adam(hashed_model.parameters(), lr=0.001)
    hashed_criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    train_model(hashed_model, hashed_optimizer, hashed_criterion, X_train, y_train)
    hashed_time = time.time() - start_time
    hashed_accuracy = measure_accuracy(hashed_model, X_train, y_train)

    # Vanilla Network
    vanilla_model = VanillaNetwork(input_dim, hidden_dim, output_dim).to(device)
    vanilla_optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=0.001)
    vanilla_criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    train_model(vanilla_model, vanilla_optimizer, vanilla_criterion, X_train, y_train)
    vanilla_time = time.time() - start_time
    vanilla_accuracy = measure_accuracy(vanilla_model, X_train, y_train)

    print(f"Hashed Network Training Time: {hashed_time:.2f}s")
    print(f"Hashed Network Accuracy: {hashed_accuracy:.2f}%")
    print(f"Vanilla Network Training Time: {vanilla_time:.2f}s")
    print(f"Vanilla Network Accuracy: {vanilla_accuracy:.2f}%")


if __name__ == "__main__":
    compare_networks()
