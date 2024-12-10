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
import torch.nn.functional as F
from torch.autograd import Variable
from adam_base import Adam

device = torch.device("cuda:0")

class HashedFC(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super(HashedFC, self).__init__()
        self.params = nn.Linear(input_dim, output_dim)
        self.K = K  # Amount of Hash Functions
        self.D = input_dim  # Input dimension
        self.L = 1  # Single hash table
        self.hash_weight = None  # Optionally provide custom hash weights
        self.num_class = output_dim  # Number of output classes
        self.init_weights(self.params.weight, self.params.bias)
        self.rehash = False
        # Activation counters and running metrics
        self.running_activations = torch.zeros(input_dim, output_dim, device='cuda:0')
        self.lsh = None


    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)
        # bias.require_gradient = False

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
        activation_mask = (activations > 0).float()
        activation_summed = torch.sum(activation_mask, dim=0)
        
        # Accumulate only activations greater than 0
        self.running_activations += activation_summed.unsqueeze(0)


    def select_representatives(self):
        """
        Select representatives based on the weighted average of activations and weights.
        """
        if self.lsh is None: 
            self.initializeLSH()
        else:
            self.rebuildLSH()

        # Get bucket indices
        bucket_indices = self.lsh.representatives()

        # Clone the weight tensor to avoid in-place modification
        new_weights = self.params.weight.clone()

        # Preallocate tensor for representatives
        device = self.params.weight.device
        representative_indices = torch.empty(len(bucket_indices), dtype=torch.long, device=device)

        # Process each bucket
        for i, bucket in enumerate(bucket_indices):
            if len(bucket) == 0:
                # Handle empty buckets by assigning a dummy value (e.g., -1)
                representative_indices[i] = -1
                continue

            if len(bucket) == 1:
                # Directly handle single-weight buckets
                single_idx = list(bucket)[0]
                row, col = single_idx // self.num_class, single_idx % self.D
                representative_indices[i] = single_idx
                new_weights[row, col] = self.params.weight[row, col]
                continue

            # Convert bucket indices to tensor
            bucket_tensor = torch.tensor(list(bucket), dtype=torch.long, device=device)

            # Convert flat indices to (row, column) pairs
            rows = bucket_tensor // self.num_class  # Row indices
            cols = bucket_tensor % self.D  # Column indices

            # Gather weights and activations for the current bucket
            bucket_activations = self.running_activations[rows, cols]  # Shape: [len(bucket)]
            bucket_weights = self.params.weight[rows, cols]  # Shape: [len(bucket)]


            """print(f"Bucket Tensor: {bucket_tensor}")
            print(f"Rows: {rows}")
            print(f"Cols: {cols}")
            print(f"Bucket Activations Shape: {bucket_activations.shape}")
            print(f"Bucket Weights Shape: {bucket_weights.shape}")"""

            # Compute weighted sum and activation sum
            weighted_sum = (bucket_weights * bucket_activations).sum()
            activation_sum = bucket_activations.sum()

            # Avoid division by zero
            if activation_sum == 0:
                representative_weight = 0
            else:
                representative_weight = weighted_sum / activation_sum

            # Find the most activated weight
            argmax_idx = bucket_activations.argmax()
            most_activated_row, most_activated_col = rows[argmax_idx], cols[argmax_idx]
            new_weights[most_activated_row, most_activated_col] = representative_weight
          


            # Store the representative index
            representative_indices[i] = most_activated_row * self.num_class + most_activated_col

        # Update the weights in the model
        self.params.weight.data = new_weights

        return representative_indices




    def prune_weights(self, representatives, input_dim):
        # Ensure `representatives` is a tensor of indices
        keep_indices = representatives

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
        #print(f"New layer after assignment: {new_layer.weight.shape}, {new_layer.bias.shape}")
        #self.he_init(new_layer.weight)  # Or he_init(new_layer.weight)
        #nn.init.zeros_(new_layer.bias)
        self.init_weights(new_layer.weight, new_layer.bias)
        # Replace the old layer
        self.params = new_layer
        self.add_module('params', self.params)
        
        #print(f"Pruned weights. Kept indices: {representatives}")"""


    def update_weights(self, input_dim):
        representatives = self.select_representatives()
        self.prune_weights(representatives, input_dim)

    def forward(self, x):
        return self.params(x)
    
  


class HashedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HashedNetwork, self).__init__()
        self.fc1 = HashedFC(input_dim, hidden_dim, K=3)
        self.fc2 = HashedFC(hidden_dim, hidden_dim, K=5)
        self.fc3 = HashedFC(hidden_dim, hidden_dim, K=4)
        self.fc4 = HashedFC(hidden_dim, output_dim, K=10)
        self.rehash = False


    def forward(self, x):
        if self.rehash:
            self.fc1.update_weights(x.shape[1])
            self.fc2.update_weights(self.fc1.num_class)
            self.fc3.update_weights(self.fc2.num_class)
            self.fc4.update_weights(self.fc3.num_class)
            self.fc1.running_activations = torch.zeros(x.shape[1], self.fc1.num_class, device='cuda:0')
            self.fc2.running_activations = torch.zeros(self.fc1.num_class, self.fc2.num_class, device='cuda:0')
            self.fc3.running_activations = torch.zeros(self.fc2.num_class, self.fc3.num_class, device='cuda:0')
            self.rehash = False


        x = F.relu(self.fc1(x))
        #activations = torch.sum(x, dim=0)
        self.fc1.accumulate_metrics(x)
        x = F.relu(self.fc2(x))
        self.fc2.accumulate_metrics(x)
        x = self.fc3(x)
        self.fc3.accumulate_metrics(x)
        x = self.fc4(x)
        self.fc4.accumulate_metrics(x)

        return x


class VanillaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

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
def train_model(model, optimizer, criterion, X_train, y_train, epochs=29, prune_every=10):
    """
    Train the model and prune hashed layers every 'prune_every' epochs.
    """
    model.train()
    for epoch in range(epochs):
        # Dynamically prune and adjust hashed layers every 'prune_every' epochs
        rehash = isinstance(model, HashedNetwork) and (epoch + 1) % prune_every == 0
        model.rehash = rehash
        

        optimizer.zero_grad()
        output = model(X_train)
        if rehash:
            print("rehashing")
            optimizer.param_groups = []
            new_params = model.fc1.params.parameters()
            optimizer.add_param_group({'params': new_params})

            new_params = model.fc2.params.parameters()
            optimizer.add_param_group({'params': new_params})

            new_params = model.fc3.params.parameters()
            optimizer.add_param_group({'params': new_params})

            new_params = model.fc4.params.parameters()
            optimizer.add_param_group({'params': new_params})


        #monitor_gradients(model)
        loss = criterion(output, y_train)
        loss.backward()
        print(f"Epoch {epoch+1} Loss: {loss}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
       

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


def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name: print(f"{name} gradient mean: {grad.mean().item()}, std: {grad.std().item()}"))

# Main comparison
def compare_networks():
    input_dim = 60
    hidden_dim = 15000
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