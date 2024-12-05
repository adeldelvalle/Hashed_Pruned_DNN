from lsh import LSH 
import torch 
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from simHash import SimHash

import clsh

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class HashedFC(nn.Module):
    def __init__(self, input_dim, output_dim, K):
        super(HashedFC, self).__init__()
        self.params = nn.Linear(input_dim, output_dim)
        self.params.bias = nn.Parameter(torch.Tensor(output_dim))  # 1D bias
        self.K = K  # Amount of Hash Functions
        self.D = input_dim  # Input dimension
        self.L = 1  # Single hash table
        self.hash_weight = None  # Optionally provide custom hash weights
        self.num_class = output_dim  # Number of output classes
        self.init_weights(self.params.weight, self.params.bias)

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

    def get_weights(self):
        # Retrieve bucket representatives
        bucket_indices = self.lsh.representatives()
        print("Bucket indices representing the weights:")
        print(bucket_indices)

        self.process_weights(bucket_indices)

    def process_weights(self, rep):
        for bucket in rep: 
            sum = 0
            count = 0
            for idx in bucket:
                weight = self.params.weight[idx]
                sum += weight 
                count += 1
            
            representative = sum/count 
            




class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = HashedFC(input_dim, hidden_dim, 10)
        self.fc2 = HashedFC(hidden_dim, output_dim, 4)

    def forward(self, x):
        x = F.relu(self.fc1.params(x))
        x = self.fc2.params(x)
        return x


# Example usage
if __name__ == "__main__":
    input_dim = 20
    hidden_dim = 500
    output_dim = 10
    batch_size = 2

    # Instantiate model
    model = SimpleMLP(input_dim, output_dim, hidden_dim)

    # Input data
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x)
    print("Output:", output)

    # Print representatives for the first layer
    print("\nRepresentatives for fc1:")
    model.fc1.get_weights()

    # Print representatives for the second layer
    print("\nRepresentatives for fc2:")
    model.fc2.get_weights()
