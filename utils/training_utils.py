import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, loader, device='cpu'):  # seconds per sample
    model.eval()
    start = time.time()
    with torch.no_grad():
        for data, _ in loader:
            model(data.to(device))
    end = time.time()
    return (end - start) / len(loader.dataset)

def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []}
    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, None, device, True)
        history['train_losses'].append(tr_loss)
        history['train_accs'].append(tr_acc)
        history['test_losses'].append(te_loss)
        history['test_accs'].append(te_acc)
        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {tr_acc:.4f}, Test Acc: {te_acc:.4f}")
    inf_time = measure_inference_time(model, test_loader, device)
    params = count_parameters(model)
    return history, inf_time, params