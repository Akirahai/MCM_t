import torch
from tqdm import tqdm


def train_model(data_loader, model, epoch, criterion, optimizer, device):
    print('Training epoch %d:' % epoch)
    num_batches = len(data_loader)
    
    total_loss = 0
    model.train()
    
    for _, (X, y) in enumerate(tqdm(data_loader)):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output.flatten(), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")