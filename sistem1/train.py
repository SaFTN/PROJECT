import ast
import json
import os
import time
import numpy as np
import torch as T
import requests
import tqdm

device = T.device('cpu')

class MNISTDataset(T.utils.data.Dataset):
    def __init__(self, image_data, label_data):
        self.x_data, self.y_data = self.load_data(image_data, label_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def load_data(self, image_data, label_data):
        images = self.load_images(image_data)
        labels = self.load_labels(label_data)
        images = images.reshape(-1, 1, 28, 28)  
        images = T.tensor(images, dtype=T.float32).to(device)
        labels = T.tensor(labels, dtype=T.int64).to(device)
        print(f"Loaded images shape {images.shape} and labels shape {labels.shape}")
        return images, labels

    def load_images(self, data):
        images = np.array(data).reshape(len(data), 28 * 28)
        return images

    def load_labels(self, data):
        labels = np.array(data)
        return labels

class NeuralNet(T.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = T.nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool = T.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = T.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = T.nn.Linear(100, 10)
  
    def forward(self, x):
        x = T.relu(self.conv1(x)) 
        x = self.pool(x)
        x = T.relu(self.conv2(x))  
        x = self.pool(x)  
        x = x.view(-1, 4 * 4 * 10)  
        x = T.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_accuracy(model, dataset):
    loader = T.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    correct_count = 0
    for data in loader:
        (pixels, labels) = data
        with T.no_grad():
            outputs = model(pixels)
        _, predictions = T.max(outputs, 1)
        correct_count += (predictions == labels).sum().item()

    accuracy = (correct_count * 1.0) / len(dataset)
    return accuracy

def fetch_initial_weights():
    url = 'http://localhost:8080/initial_weights'
    response = requests.get(url)
    if response.status_code == 200:
        weights = response.json()
        return weights
    elif response.status_code == 404:
        print("No initial weights available.")
        return None
    else:
        print(f"Error fetching initial weights: {response.status_code}, {response.text}")
        return None

def apply_weights_to_model(model, weights):
    model.conv1.weight.data = T.tensor(weights['conv1_weight'], dtype=T.float32).to(device)
    model.conv1.bias.data = T.tensor(weights['conv1_bias'], dtype=T.float32).to(device)
    model.conv2.weight.data = T.tensor(weights['conv2_weight'], dtype=T.float32).to(device)
    model.conv2.bias.data = T.tensor(weights['conv2_bias'], dtype=T.float32).to(device)
    model.fc1.weight.data = T.tensor(weights['fc1_weight'], dtype=T.float32).to(device)
    model.fc1.bias.data = T.tensor(weights['fc1_bias'], dtype=T.float32).to(device)
    model.fc2.weight.data = T.tensor(weights['fc2_weight'], dtype=T.float32).to(device)
    model.fc2.bias.data = T.tensor(weights['fc2_bias'], dtype=T.float32).to(device)

def upload_weights_to_server(weights):
    url = 'http://localhost:8080/weights'  
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(weights), headers=headers)
    if response.status_code == 200:
        print("Weights sent successfully")
    else:
        print(f"Error sending weights: {response.status_code}, {response.text}")

def convert_model_weights_to_dict(model):
    weights = {
        'conv1_weight': model.conv1.weight.detach().cpu().numpy().tolist(),
        'conv1_bias': model.conv1.bias.detach().cpu().numpy().tolist(),
        'conv2_weight': model.conv2.weight.detach().cpu().numpy().tolist(),
        'conv2_bias': model.conv2.bias.detach().cpu().numpy().tolist(),
        'fc1_weight': model.fc1.weight.detach().cpu().numpy().tolist(),
        'fc1_bias': model.fc1.bias.detach().cpu().numpy().tolist(),
        'fc2_weight': model.fc2.weight.detach().cpu().numpy().tolist(),
        'fc2_bias': model.fc2.bias.detach().cpu().numpy().tolist()
    }
    return weights

def save_model_parameters_to_file(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\n")
            f.write(f"{param.detach().cpu().numpy().tolist()}\n")

def load_model_parameters_from_file(model, filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        param_dict = {}
        for i in range(0, len(lines), 2):
            name = lines[i].strip()
            param = ast.literal_eval(lines[i+1].strip())
            param_dict[name] = T.tensor(param, dtype=T.float32).to(device)

        model.load_state_dict(param_dict)

def main():
    np.random.seed(1)
    T.manual_seed(1)


    print("\nFetching MNIST training dataset from GoLang HTTP Server")

    num_items = 10 
    url = f'http://localhost:8080/mnist_data?num={num_items}'

    max_retries = 15
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            print("Server is not up yet, retrying...")
        time.sleep(1)
    else:
        print("Server did not respond within 15 seconds. Exiting.")
        return

    if response.status_code != 200:
        print(f"Error fetching data from server: {response.status_code}")
        return

    data = response.json()
    image_data = data['images']
    label_data = data['labels']

    training_dataset = MNISTDataset(image_data, label_data)

    batch_size = 50
    train_loader = T.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    print("\nStarting training script")
    net = NeuralNet().to(device)

    initial_weights = fetch_initial_weights()
    if initial_weights:
        apply_weights_to_model(net, initial_weights)
        print("\nLoaded existing weights")
    else:
        print("\nInitializing random weights")

    max_epochs = 2
    epoch_log_interval = 1
    learning_rate = 0.001
    
    loss_function = T.nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(net.parameters(), lr=learning_rate)
    
    print(f"\nbatch_size = {batch_size}")
    print(f"loss_function = {loss_function}")
    print("optimizer = Adam")
    print(f"max_epochs = {max_epochs}")
    print(f"learning_rate = {learning_rate}")

    print("\nStarting training")
    net.train()

    for epoch in range(max_epochs):
        epoch_loss = 0
        for (X, y) in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % epoch_log_interval == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}")
        if (epoch+1) % 5 == 0:
            acc = compute_accuracy(net, training_dataset)
            print(f"Accuracy on training set: {acc * 100:.2f}%")

    net.eval()
    if True:
        acc = compute_accuracy(net, training_dataset)
        print(f"Accuracy on training set: {acc * 100:.2f}%")

    upload_weights_to_server(convert_model_weights_to_dict(net))
    save_model_parameters_to_file(net, "neural_model/mnist_model.txt")

    print("\nEnd of training script")

if __name__ == "__main__":
    main()
