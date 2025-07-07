import os
import time
import pandas as pd
import torch

from utils.comparison_utils import get_cifar_loaders, get_mnist_loaders
from models.fc_models import FC, DeepFC
from models.cnn_models import  SimpleCNN, CNNWithResidual, RegularizedResNet
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history, plot_confusion_matrix


def mnist():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 5
    BATCH_SIZE = 64

    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)

    models = {
        "FC": FC(input_size=784),
        "SimpleCNN": SimpleCNN(),
        "CNNWithResidual": CNNWithResidual()
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        history, inference_time, param_count = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device=device)
        end_time = time.time()
        training_time = end_time - start_time


        results.append({
            "Model": name,
            "Train Acc": history['train_accs'][-1],
            "Test Acc": history['test_accs'][-1],
            "Train Loss": history['train_losses'][-1],
            "Test Loss": history['test_losses'][-1],
            "Parameters": param_count,
            f"Training Time ({EPOCHS} epochs)": training_time,
            "Inference Time (per sample)": inference_time,
            "History": history
        })

        plot_training_history(history, f"results/mnist_comp/mnist_comp_{name}.png")

    df = pd.DataFrame([{k: v for k, v in res.items() if k != 'History'} for res in results])
    print(df)
    df.to_csv('results/mnist_comp/mnist_comp_results.csv', index=False)

def cifar():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 2
    BATCH_SIZE = 128

    # CIFAR-10 загрузка
    train_loader, test_loader = get_cifar_loaders(batch_size=BATCH_SIZE)
    classes = train_loader.dataset.dataset.classes

    models = {
        'DeepFC': DeepFC(),
        'ResCNN': CNNWithResidual(),
        'RegResCNN': RegularizedResNet()
    }

    results = []
    save_dir = 'results/cifar_comp'
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        print(f'Training {name}...')
        history,inf_time, params = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device=device)
        # распаковка
        train_losses = history['train_losses']
        train_accs = history['train_accs']
        test_losses = history['test_losses']
        test_accs = history['test_accs']



        # confusion matrix
        plot_confusion_matrix(model, test_loader, device, classes, name)

        results.append({
            'Model': name,
            'Train Acc': train_accs[-1],
            'Test Acc': test_accs[-1],
            'Train Loss': train_losses[-1],
            'Test Loss': test_losses[-1],
            'Parameters': params,
            'Inference Time (s/sample)': inf_time
        })

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(save_dir, 'cifar10_comparison_results.csv'), index=False)

if __name__ == '__main__':
    mnist()
    cifar()