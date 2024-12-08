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
        #self.init_weights(self.params.weight, self.params.bias)
        self.rehash = False
        # Activation counters and running metrics
        self.running_activations = torch.zeros(output_dim, device='cuda:0')
        self.lsh = None


    def init_weights(self, weight, bias):
        nn.init.kaiming_uniform_(weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(bias)

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

            # Convert bucket indices to tensor
            bucket_tensor = torch.tensor(list(bucket), dtype=torch.long, device=device)

            # Gather weights and activations for the current bucket
            bucket_activations = self.running_activations[bucket_tensor]
            bucket_weights = self.params.weight[bucket_tensor]

            # Compute weighted sum of weights and activations
            weighted_sum = (bucket_weights * bucket_activations.unsqueeze(1)).sum(dim=0)
            num_weights = len(bucket)

            # Calculate representative weight
            representative_weight = weighted_sum / num_weights

            # Find the index of the most activated weight
            most_activated_idx = bucket_tensor[bucket_activations.argmax()]
            new_weights[most_activated_idx] = representative_weight

            # Store the representative index
            representative_indices[i] = most_activated_idx

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
    
    def he_init(self, weight):
        if weight.dim() > 1:  # Ensure it's not applied to bias (1D tensor)
            nn.init.kaiming_uniform_(weight, mode='fan_in', nonlinearity='relu')

    
    def accumulate_metrics(self, input_dim, activations):
        # Binary mask: 1 if activated, 0 otherwise
        activation_summed = torch.sum(activations, dim=0)
        activation_mask = (activation_summed > 0).float()
        # Accumulate only activations greater than 0
        self.running_activations += activation_mask
        return self.rehash_or_not(input_dim, activations)


    def rehash_or_not(self, input_dim, activations):
        #print("current input dim:", self.D)
        entropy = self.calc_activation_entropy(activations)
        print(f"entropy: {entropy}")

        if entropy >= 0.975:
            self.update_weights(input_dim)
            self.running_activations = torch.zeros(self.num_class, device='cuda:0')
            return True
        return False