from lsh import LSH
import torch
import numpy as np
import math
import torch.nn as nn
from simHash import SimHash
import time

device = torch.device("cuda:0")

class HashedFC(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super(HashedFC, self).__init__()
        self.params = nn.Linear(input_dim, output_dim)
        self.K = 5  # Amount of Hash Functions
        self.D = input_dim  # Input dimension
        self.L = 1  # Single hash table
        self.hash_weight = None  # Optionally provide custom hash weights
        self.num_class = output_dim  # Number of output classes
        self.init_weights(self.params.weight, self.params.bias)
        self.rehash = False
        # Activation counters and running metrics
        self.running_activations = torch.zeros(input_dim, output_dim, device='cuda:0')
        self.lsh = None
        self.prev_weight_mean = None


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
        #print("new input dim:", self.D)

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


    def calc_activation_entropy(self, activations):
        """
        Calculate the normalized entropy of activations.

        Args:
            activations (torch.Tensor): Activations of shape (batch_size, num_neurons).

        Returns:
            float: Normalized activation entropy between 0 and 1.
        """
        # Flatten activations into a single dimension
        flat_activations = activations.view(-1)
        
        # Normalize to probabilities
        probs = torch.abs(flat_activations) / torch.sum(torch.abs(flat_activations))
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()  # Add epsilon to avoid log(0)
        
        # Normalize by maximum entropy
        max_entropy = torch.log(torch.tensor(probs.numel(), dtype=torch.float32)).item()
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy


    def forward(self, x):
        return self.params(x)

    
    def accumulate_metrics(self, input_dim, activations):
        # Binary mask: 1 if activated, 0 otherwise
        activation_mask = (activations > 0).float()
        activation_summed = torch.sum(activation_mask, dim=0)
        
        # Accumulate only activations greater than 0
        self.running_activations += activation_summed.unsqueeze(0)
        return self.rehash_or_not(input_dim, activations)


    def rehash_or_not(self, input_dim, activations):
        #print("current input dim:", self.D)
        entropy = self.calc_activation_entropy(activations)
        weight_change = self.calc_weight_change()
        print(f"entropy: {entropy}")


        #print(weight_change)
        if entropy >= 0.965:
            self.update_weights(input_dim)
            self.running_activations = torch.zeros(input_dim, self.num_class, device='cuda:0')
            return True
        return False
    
    def calc_weight_change(self):
    # Initialize `prev_weight_mean` if it doesn't exist
        if self.prev_weight_mean == None:
            self.prev_weight_mean = torch.mean(self.params.weight.data).item()
            return 0 

        # Compute the current mean of weights
        current_mean = torch.mean(self.params.weight.data)

        # Calculate the squared difference between the current and previous mean
        weight_change = (self.prev_weight_mean - current_mean) ** 2

        # Update the `prev_weight_mean` for the next computation
        self.prev_weight_mean = current_mean

        return weight_change.item()  # Return a scalar