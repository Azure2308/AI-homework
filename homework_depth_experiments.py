import time
import torch

from utils.dataset import get_mnist_dataloaders, get_cifar_dataloaders
from utils.models import FullyConnectedModel
from utils.trainer import count_parameters, train_model
from utils.visualization import plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Ваш девайс',device)

### 1.1 Сравнение моделей разной глубины (15 баллов)
# Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
# - 2 слоя (1 скрытый)
# - 3 слоя (2 скрытых)
# - 5 слоев (4 скрытых)
# - 7 слоев (6 скрытых)
#
# Для каждого варианта:
# - Сравните точность на train и test
# - Визуализируйте кривые обучения
# - Проанализируйте время обучения

def depth_experiments():
    datasets = ["MNIST", "CIFAR"]
    dataloader_getters = {
        "MNIST": get_mnist_dataloaders,
        "CIFAR": get_cifar_dataloaders
    }

    # Конфигурации моделей
    configs = [
        {
            "name": "1 layer",
            "layers": [{"type": "linear", "size": 10}]
        },
        {
            "name": "2 layers",
            "layers": [
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "3 layers",
            "layers": [
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "5 layers",
            "layers": [
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "7 layers",
            "layers": [
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 32},
                {"type": "relu"},
                {"type": "linear", "size": 16},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "Dropout + BatchNorm layers",
            "layers": [
                {"type": "linear", "size": 512},
                {"type": "batch_norm"},
                {"type": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "dropout", "rate": 0.1},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
    ]

    for dataset_name in datasets:
        print(f"\nДатасет: {dataset_name}\n")
        train_loader, test_loader = dataloader_getters[dataset_name](batch_size=1024)

        for cfg in configs:
            print(f"Конфиг: {cfg['name']}")

            start_time = time.time()

            # Два датасета с разными входными данными
            input_size = 784 if dataset_name == "MNIST" else 3072

            model = FullyConnectedModel(input_size=input_size, num_classes=10, layers=cfg["layers"]).to(device)

            params_count = count_parameters(model)
            print(f"Параметры: {params_count}")

            history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Time taken to train: {training_time:.2f} seconds")

            plot_training_history(history, f'plots/{dataset_name}_{cfg["name"].replace(" ", "_")}')


if __name__ == "__main__":
    depth_experiments()