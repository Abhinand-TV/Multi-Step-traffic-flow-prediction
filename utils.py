import torch


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)