import matplotlib.pyplot as plt
import os
import seaborn as sns

import torch
from sklearn.metrics import confusion_matrix


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, loader, device, classes, tag):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Conf Matrix: {tag}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    os.makedirs('results/cifar_comp', exist_ok=True)
    plt.savefig(f'results/cifar_comp/cifar_comp_{tag}.png')
    plt.close()