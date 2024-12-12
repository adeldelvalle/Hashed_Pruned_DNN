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
from fvcore.nn import FlopCountAnalysis
import utils

import matplotlib.pyplot as plt

device = torch.device("cuda:0")
from hashedFC import HashedFC


class HashedNetwork(nn.Module):
    """ Class for the Hash Network. It creates several sub-modules from from the HashedFC
    class, which have all the functions to update weights, accumulate metrics, 
    initialize LSH, and more. 
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HashedNetwork, self).__init__()
        self.fc1 = HashedFC(input_dim, hidden_dim, K=3)
        self.fc2 = HashedFC(hidden_dim, hidden_dim, K=5)
        self.fc3 = HashedFC(hidden_dim, hidden_dim, K=4)
        self.fc4 = HashedFC(hidden_dim, output_dim, K=10)
        self.rehash = False


    def forward(self, x):
        if self.rehash:
            # ------------------- Rehashing and updating weights --------------- #
            self.fc1.update_weights(x.shape[1])
            self.fc2.update_weights(self.fc1.num_class)
            self.fc3.update_weights(self.fc2.num_class)
            self.fc4.update_weights(self.fc3.num_class)

            # ------------------ Resetting activations tensors for the new rehashed weights ------------- #
            self.fc1.running_activations = torch.zeros(x.shape[1], self.fc1.num_class, device='cuda:0')
            self.fc2.running_activations = torch.zeros(self.fc1.num_class, self.fc2.num_class, device='cuda:0')
            self.fc3.running_activations = torch.zeros(self.fc2.num_class, self.fc3.num_class, device='cuda:0')
            self.rehash = False


        x = F.relu(self.fc1(x))
        self.fc1.accumulate_metrics(x)  # Accumulate activations for the weighted average
        x = F.relu(self.fc2(x))
        self.fc2.accumulate_metrics(x)
        x = F.relu(self.fc3(x))
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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# Train function
def train_model(model, optimizer, criterion, X_train, y_train, epochs=29, prune_every=10):
    """
    Train the model and prune hashed layers every 'prune_every' epochs.
    """
    loss_history = []

    model.train()
    for epoch in range(epochs):
        # Dynamically prune and adjust hashed layers every 'prune_every' epochs
        rehash = isinstance(model, HashedNetwork) and (epoch + 1) % prune_every == 0
        model.rehash = rehash
        

        optimizer.zero_grad()
        output = model(X_train)
        if rehash:
            # -------------------- Temporal solution while thinking of how to modify the optimizer/gradient parameters in place  ----------- #
            # --------------------  We clean the param groups because the optimizer will try to optimize the pruned params ----------- #

            # - To-do / Enhancement: Do rehashing layer-wise instead of the full network. 

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
        loss_history.append(loss.item())

        optimizer.step()
    return loss_history
       

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
    """ Method to compare networks. It creates an instance of every network, and trains them
     with the same parameters, optimizers, struture, and data.  """
    input_dim = 60
    hidden_dim = 15000
    output_dim = 2

    # Generate data
    X, y = utils.generate_synthetic_data(n_samples=10000, n_features=input_dim, n_classes=output_dim)
    #X, y = get_higgs_small_dataset()

    
    #X_train, y_train = X.to(device), y.to(device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data 
    

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long).to(device), torch.tensor(y_test, dtype=torch.long).to(device)

    # Hashed Network
    hashed_model = HashedNetwork(X.shape[1], hidden_dim, output_dim).to(device)
    hashed_optimizer = torch.optim.Adam(hashed_model.parameters(), lr=0.001)
    hashed_criterion = nn.CrossEntropyLoss()
    #wandb.init(project="hashed-vs-vanilla", name="HashedNetworkAnalysis")

    start_time = time.time()

    # --------------------------- Training of Hashed Neural Network --------------------------- #
    
    hashed_loss_history = train_model(hashed_model, hashed_optimizer, hashed_criterion, X_train, y_train)
    hashed_time = time.time() - start_time
    hashed_accuracy = measure_accuracy(hashed_model, X_train, y_train)
    hashed_test_accuracy = measure_accuracy(hashed_model, X_test, y_test)
    #wandb.finish()


    # --------------------------- Training of Vanilla Neural Network --------------------------- #

    vanilla_model = VanillaNetwork(X.shape[1], hidden_dim, output_dim).to(device)
    vanilla_optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=0.001)
    vanilla_criterion = nn.CrossEntropyLoss()
    #wandb.init(project="hashed-vs-vanilla", name="VanillaNetworkAnalysis")

    start_time = time.time()
    
    vanilla_loss_history = train_model(vanilla_model, vanilla_optimizer, vanilla_criterion, X_train, y_train)
    vanilla_time = time.time() - start_time
    vanilla_accuracy = measure_accuracy(vanilla_model, X_train, y_train)
    vanilla_test_accuracy = measure_accuracy(vanilla_model, X_test, y_test)
    #wandb.finish()

    # -------------------------- Training Finished, print stats -------------------------------- #

    print(f"Hashed Network Training Time: {hashed_time:.2f}s")
    print(f"Hashed Network Accuracy: {hashed_accuracy:.2f}%")
    print(f"Hashed Network - Test Accuracy: {hashed_test_accuracy:.2f}%")

    flops = FlopCountAnalysis(hashed_model, torch.randn(1, X.shape[1]).to(device))
    print(f"Hashed Network FLOPS: {flops.total()}")
    print_total_parameters(hashed_model, model_name="Hashed Network")

    flops = FlopCountAnalysis(vanilla_model, torch.randn(1, X.shape[1]).to(device))

    print(f"Vanilla Network Training Time: {vanilla_time:.2f}s")
    print(f"Vanilla Network Accuracy: {vanilla_accuracy:.2f}%")
    print(f"Vanilla Network - Test Accuracy: {vanilla_test_accuracy:.2f}%")

    print(f"Vanilla Network FLOPS: {flops.total()}")
    print_total_parameters(vanilla_model, model_name="Vanilla Network")


    # Plotting
    utils.plot_results(hashed_loss_history, vanilla_loss_history)
    utils.plot_layerwise_weight_distribution(hashed_model, "HashedNN")
    utils.plot_layerwise_weight_distribution(vanilla_model, "VanillaNN")



def print_total_parameters(model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} - Total Parameters: {total_params:,}")
    print(f"{model_name} - Trainable Parameters: {trainable_params:,}")



if __name__ == "__main__":
    compare_networks()
