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
from adam_base import Adam
from hashedFC import HashedFC

device = torch.device("cuda:0")

class HashedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HashedNetwork, self).__init__()
        self.fc1 = HashedFC(input_dim, hidden_dim, K=3)
        self.fc2 = HashedFC(hidden_dim, hidden_dim, K=5)
        self.fc3 = HashedFC(hidden_dim, hidden_dim, K=4)
        self.fc4 = HashedFC(hidden_dim, output_dim, K=4)
        self.rehash = False
        self.update_layers = {'fc1': False, 'fc2': False, 'fc3': False}
        self.indexes = {'fc1':'fc2', 'fc2':'fc3', 'fc3':'fc4'}
      


    def forward(self, x):
        """if self.rehash:
            self.fc1.update_weights(x.shape[1])
            self.fc2.update_weights(self.fc1.num_class)
            self.fc3.update_weights(self.fc2.num_class)
            self.fc4.update_weights(self.fc3.num_class)
            self.fc1.running_activations = torch.zeros(self.fc1.num_class, device='cuda:0')
            self.fc2.running_activations = torch.zeros(self.fc2.num_class, device='cuda:0')
            self.fc3.running_activations = torch.zeros(self.fc3.num_class, device='cuda:0')
            self.rehash = False"""

        input_dim = x.shape[1]
        x = F.relu(self.fc1(x))
        self.update_layers['fc1'] = self.fc1.accumulate_metrics(input_dim, x)
        x = F.relu(self.fc2(x))
        self.update_layers['fc2'] = self.fc2.accumulate_metrics(self.fc1.num_class, x)
        x = self.fc3(x)
        self.update_layers['fc3'] = self.fc3.accumulate_metrics(self.fc2.num_class, x)
        x = self.fc4(x)

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
        #rehash = isinstance(model, HashedNetwork) and (epoch + 1) % prune_every == 0
        #model.rehash = rehash
        

        optimizer.zero_grad()
        output = model(X_train)
        #if rehash:

        if isinstance(model, HashedNetwork):
            for layer_name, rehash in model.update_layers.items():
                if rehash:
                    # Dynamically get the layer using getattr
                    print(f"hashing {layer_name}")
                    layer = getattr(model, layer_name)
                    next_layer = getattr(model, model.indexes[layer_name])

                    #print(f"{layer} is rehashing!")
                    # Get the new parameters from the layer
                    new_params = layer.params.parameters()
                    # Update the optimizer
                    update_optimizer_param_group(optimizer, layer_name, new_params)
                    next_name = model.indexes[layer_name]
                    model.update_layers[layer_name] = False
                    if next_name == 'fc4' or not model.update_layers[next_name]:
                        print(next_name)
                        recreate_next_layer(layer, next_layer, optimizer)


        """optimizer.param_groups = []
        new_params = model.fc1.params.parameters()
        optimizer.add_param_group({'params': new_params})

        new_params = model.fc2.params.parameters()
        optimizer.add_param_group({'params': new_params})

        new_params = model.fc3.params.parameters()
        optimizer.add_param_group({'params': new_params})

        new_params = model.fc4.params.parameters()
        optimizer.add_param_group({'params': new_params})"""


        #monitor_gradients(model)
        loss = criterion(output, y_train)
        loss.backward()
        print(f"Epoch {epoch+1} Loss: {loss}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()


def recreate_next_layer(prev_layer, next_layer, optimizer):
    """
    Recreate the next layer with updated dimensions based on the output of the previous layer.

    Args:
        prev_layer: The rehashed layer (e.g., model.fc1).
        next_layer: The subsequent layer to be recreated (e.g., model.fc2).
    """
    new_input_dim = prev_layer.num_class  # Updated input dimensions

    # Save old weights and biases
    old_weights = next_layer.params.weight.data[:, :new_input_dim].clone()
    old_biases = next_layer.params.bias.data.clone()

    # Recreate the next layer
    new_layer = nn.Linear(new_input_dim, next_layer.num_class).to(next_layer.params.weight.device)
    new_layer.weight.data[:old_weights.size(0), :] = old_weights
    new_layer.bias.data[:old_biases.size(0)] = old_biases

    # Replace the next layer with the new one
    next_layer.params = new_layer
    next_layer.D = new_input_dim

    new_params = next_layer.params.parameters()
    optimizer.add_param_group({'params': new_params})


def slice_next_layer_weights(prev_layer, next_layer, optimizer):
    """
    Slices the weights of the next layer to match the new output dimensions of the previous layer
    and updates the optimizer state for the sliced weights.

    Args:
        prev_layer: The rehashed layer (e.g., model.fc1).
        next_layer: The subsequent layer to be sliced (e.g., model.fc2).
        optimizer: The optimizer being used for training.
    """
    original_tensor = next_layer.params.weight.clone()  # Reference to the original weight tensor
    new_input_dim = prev_layer.num_class  # New input dimension based on the rehashed previous layer
    
    with torch.no_grad():
        # Slice the weights to match the new input dimension
        next_layer.params.weight.data = next_layer.params.weight.data[:, :new_input_dim]

    sliced_tensor = next_layer.params.weight


    # Update the input dimension of the next layer
    next_layer.D = new_input_dim

    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param is original_tensor:
                group['params'][i] = next_layer.params.weight
                return

    device = sliced_tensor.device
    dtype = sliced_tensor.dtype

    if original_tensor not in optimizer.state:
        print(f"State not found for tensor: {original_tensor.shape}. Initializing state.")
        optimizer.state[sliced_tensor] = {
            'step': torch.tensor(0.0, device=device, dtype=torch.float32),
            'exp_avg': torch.zeros_like(sliced_tensor, device=device, dtype=dtype),
            'exp_avg_sq': torch.zeros_like(sliced_tensor, device=device, dtype=dtype),
        }
        return

    state = optimizer.state.pop(original_tensor)

    # Resize and move state tensors to the new tensor
    if 'exp_avg' in state:
        state['exp_avg'] = state['exp_avg'][:, :sliced_tensor.size(1)].to(device=device, dtype=dtype)
    if 'exp_avg_sq' in state:
        state['exp_avg_sq'] = state['exp_avg_sq'][:, :sliced_tensor.size(1)].to(device=device, dtype=dtype)

    optimizer.state[sliced_tensor] = state



def update_optimizer_param_group(optimizer, layer_name, new_params):
    """
    Updates the parameters in the optimizer's parameter group for a specific layer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        layer_name (str): The name of the layer being updated (optional if used for clarity).
        new_params (iterable): The new parameters for the layer.
    """
    # Find the first parameter group (assuming each layer has its own group)
    for group in optimizer.param_groups:
        if layer_name in group.get('name', ''):  # Optional: check the layer name
            # Replace parameters in this group
            group['params'] = new_params

            # Clear the optimizer state for the new parameters
            for p in new_params:
                if p in optimizer.state:
                    optimizer.state.pop(p)
            break



       

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
