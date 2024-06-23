import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import psutil
import os

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory in MB

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Model definition
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(14 * 14, 64, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

net = RNNModel()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=15e-3)

# Function to calculate loss and accuracy
def loss_and_accuracy(model, data_loader):
    model.eval()
    with torch.no_grad():
        for (x, y) in data_loader:
            x = x.view(-1, 4, 196)  # Reshape for RNN
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = (y_pred.argmax(1) == y).float().mean().item() * 100
    return loss.item(), round(acc, 2)

# Training the model
train_log = []
settings = {
    'eta': 15e-3,
    'epochs': 5,
}

start_time = time.time()
start_memory = get_memory_usage()

for epoch in range(settings['epochs']):
    net.train()
    for (x, y) in train_loader:
        x = x.view(-1, 4, 196)  # Reshape for RNN
        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = loss_and_accuracy(net, train_loader)
    test_loss, test_acc = loss_and_accuracy(net, test_loader)
    print(f'Epoch {epoch+1}, Train Accuracy: {train_acc}%, Test Accuracy: {test_acc}%')
    train_log.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc})

end_time = time.time()
end_memory = get_memory_usage()

print(f'Total training time: {end_time - start_time:.2f} seconds')
print(f'Memory used: {end_memory - start_memory:.2f} MB')

# Checking some predictions
x1, y1 = next(iter(train_loader))
x1 = x1.view(-1, 4, 196)
y1_pred = net(x1)