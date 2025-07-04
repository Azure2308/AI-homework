import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from utils.dataset import get_mnist_dataloaders
from utils.models import FullyConnectedModel
from utils.trainer import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 3.1 Сравнение техник регуляризации (15 баллов)
# Исследуйте различные техники регуляризации:
# - Без регуляризации
# - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
# - Только BatchNorm
# - Dropout + BatchNorm
# - L2 регуляризация (weight decay)
#
# Для каждого варианта:
# - Используйте одинаковую архитектуру
# - Сравните финальную точность
# - Проанализируйте стабильность обучения
# - Визуализируйте распределение весов

# Общая архитектура модели
config = [
    {"type": "linear", "size": 512},
    {"type": "relu"},
    {"type": "linear", "size": 256},
    {"type": "relu"},
    {"type": "linear", "size": 128},
    {"type": "relu"},
    {"type": "linear", "size": 10}
]

# Чтобы не писать отдельный конфиг к разным методам, я сделал через срезы =)
regularization_method = [
    {"name": "Без регуляризации", "config": config.copy()},
    {"name": "Dropout (с=0.1)",
     "config": config[:3] + [{"type": "dropout", "rate": 0.1}] + config[3:]},
    {"name": "Dropout (с=0.3)",
     "config": config[:3] + [{"type": "dropout", "rate": 0.3}] + config[3:]},
    {"name": "Dropout (с=0.5)",
     "config": config[:3] + [{"type": "dropout", "rate": 0.5}] + config[3:]},
    {"name": "BatchNorm", "config": config[:3] + [{"type": "batch_norm"}] + config[3:]},
    {"name": "Dropout+BatchNorm",
     "config": config[:3] +
               [{"type": "dropout", "rate": 0.3},
                {"type": "batch_norm"}]
               + config[3:]
     },
    {"name": "L2 Регуляризация", "config": config.copy()}
]

def regularization_experiments():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    data = []
    for method in regularization_method:
        print(f"Техника: {method['name']}")

        model = FullyConnectedModel(input_size=784, num_classes=10, layers=method["config"]).to(device)

        optimizer = optim.Adam(model.parameters()) if method["name"] != "L2 Регуляризация" else optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device), optimizer=optimizer)

        final_accuracy = max(history["test_accs"])
        weights_distribution = get_weight_statistics(model)

        data.append({
            "method": method["name"],
            "final_accuracy": final_accuracy,
            "weights": weights_distribution
        })

    df = pd.DataFrame(data)
    df = df[["method", "final_accuracy"]]
    df.columns = ["Метод", "Точность"]

    print("\nРезультаты сравнения методов регуляризации:")
    print(df.to_markdown(index=False, tablefmt="grid"))

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[d["weights"] for d in data])
    plt.xticks(range(len(data)), [d["method"] for d in data], rotation=45)
    plt.title("Распределение весов")
    plt.ylabel("Значение веса")
    plt.tight_layout()
    plt.savefig("plots/weight_distribution")
    plt.show()

# Для получения распределения весов модели
def get_weight_statistics(model):
    weights = []
    for p in model.parameters():
        if p.dim() > 1:  # Берём только веса, игнорируя смещения
            weights.extend(p.detach().cpu().numpy().flatten())
    return weights

if __name__ == "__main__":
    regularization_experiments()
