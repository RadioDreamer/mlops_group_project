import matplotlib.pyplot as plt
from torch import cuda, device, nn, optim, save
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import FakeArtClassifier

DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")


def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 32) -> None:
    """Train a model on the CIFAKE dataset"""
    # TODO: the prints bellow will become loguru aftewards
    print("Training day and night")
    print(f"lr: {lr}, epochs: {epochs}, batch_size: {batch_size}")

    model = FakeArtClassifier().to(DEVICE)

    train_set, _ = cifake()
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    save(model.state_dict(), "models/model.pth")

    # make a nice statistic plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

    # for epoch in range(2):
    #    model.train()
    #    for batch_idx, (images, labels) in enumerate(train_loader):
    #        images, labels = images.to(device), labels.to(device)

    #        optimizer.zero_grad()
    #        outputs = model(images)
    #        loss = criterion(outputs, labels)
    #        loss.backward()
    #        optimizer.step()

    #        if batch_idx % 50 == 0:
    #            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train()
