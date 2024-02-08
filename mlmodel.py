from cProfile import label
from pickletools import optimize
from pyexpat import model
from random import shuffle
from sympy import root
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image

device = torch.device('cpu')
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate=0.001

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.137),(0.3081))]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(torch.device('cpu'))  # Replace 'cpu' with the appropriate device

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps= len(train_data_loader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_data_loader):
        #100,1,28,28
        images=images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        outputs=model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i-1) % 100 ==0:
            print(f'Epochs[{epoch+1}/{num_epochs},Step{i+1}/{n_total_steps,}]Loss:{loss.item():.4f}')

with torch.no_grad():
    n_correct=0
    n_samples=0

    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs= model(images)
        _,predicted=torch.max(outputs.data,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        acc =100.0 * n_correct/n_samples
        print(f"Accuracy:{acc}")

torch.save(model.state_dict(),"mnist_ffn.pth")
