import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ParameterGrid
from utils.dataset import get_mnist_dataloaders
from utils.models import FullyConnectedModel
from utils.trainer import count_parameters, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2.1 Сравнение моделей разной ширины (15 баллов)
# Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
# - Средние слои: [256, 128, 64]
# - Широкие слои: [1024, 512, 256]
# - Очень широкие слои: [2048, 1024, 512]
#
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров

def width_experiments():
    configs = [
        {
            "name": "Узкие слои",
            "layers": [
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
            "name": "Средние слои",
            "layers": [
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
            "name": "Широкие слои",
            "layers": [
                {"type": "linear", "size": 1024},
                {"type": "relu"},
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "Очень широкие слои",
            "layers": [
                {"type": "linear", "size": 2048},
                {"type": "relu"},
                {"type": "linear", "size": 1024},
                {"type": "relu"},
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        }
    ]

    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    for cfg in configs:
        print(f"\nМодель: {cfg['name']}\n")

        start_time = time.time()

        model = FullyConnectedModel(input_size=784, num_classes=10, layers=cfg["layers"]).to(device)
        params_count = count_parameters(model)
        print(f"Параметры: {params_count}")
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Time taken to train: {training_time:.2f} seconds")

### 2.2 Оптимизация архитектуры (10 баллов)
# Найдите оптимальную архитектуру:
# - Используйте grid search для поиска лучшей комбинации
# - Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная)
# - Визуализируйте результаты в виде heatmap

# Преобразует ширину слоев в стандартизированный формат
def layers_config(widths):
    return [
        {"type": "linear", "size": widths[0]}, {"type": "relu"},
        {"type": "linear", "size": widths[1]}, {"type": "relu"},
        {"type": "linear", "size": widths[2]}, {"type": "relu"},
        {"type": "linear", "size": 10}
    ]

# Оценивает архитектуру по Accuracy и Параметрам
def evaluate_architectures(train_loader, test_loader, width_scheme):
    start_time = time.time()

    model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers_config(width_scheme)).to(device)
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

    accuracy = max(history["test_accs"])
    training_time = time.time() - start_time
    param_count = count_parameters(model)

    return {
        "width_scheme": tuple(width_scheme),  # преобразуем список в кортеж!
        "accuracy": accuracy,
        "param_count": param_count,
        "training_time": training_time
    }


def find_optimal_width():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=1024)

    parameter_grid = {
        "width_scheme": [
            [64, 128, 256],
            [256, 128, 64],
            [128, 128, 128],
            [256, 256, 256],
            [512, 512, 512]
        ]
    }
    data = []

    # Запускаем grid search
    for params in ParameterGrid(parameter_grid):
        data.append(evaluate_architectures(train_loader, test_loader, params["width_scheme"]))

    df_results = pd.DataFrame(data)
    pivot_data = df_results.pivot(index='width_scheme', columns='param_count', values='accuracy')

    # Heatmap Точность/Параметры
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Heatmap width experiments")
    plt.savefig("plots/heatmap_optimal_width")
    plt.show()


if __name__ == "__main__":
    width_experiments() # 2.1
    find_optimal_width() #2.2