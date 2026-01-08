import torch
from torch import nn
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import Model


def train():
    # Load data
    train_set, test_set = cifake("data/processed")
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train()
