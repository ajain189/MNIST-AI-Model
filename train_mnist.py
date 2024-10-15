import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cpu")
model = model.to(device)

epochs = 1
for epoch in range(epochs):
  model.train()
  total_loss = 0
  for batch_idx, (data, target)in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    ouputs = model(data)

    loss = loss_fn(ouputs, target)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()


    if batch_idx % 5 == 4:
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss}")
        total_loss = 0.0

    torch.save(model.state_dict(), f"mnist_{epoch}.pth")

print("Done")

torch.save(model.state_dict(), "mnist_fully_trained.pth")