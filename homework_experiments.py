import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


from homework_datasets import MushroomDataset
from homework_model_log_modification import LogisticRegressionTorch, train, evaluate

### 3.1 Исследование гиперпараметров (10 баллов)
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц


# Константы
DATA_PATH = './data/mushrooms.csv'

# Разделение данных
full_ds = MushroomDataset(DATA_PATH)
n = len(full_ds)
test_n = int(0.2 * n)
val_n = int(0.1 * (n - test_n))
train_n = n - test_n - val_n
train_ds, val_ds, test_ds = random_split(full_ds, [train_n, val_n, test_n])

# 3.1 Гиперпараметры
learning_rates = [1e-3, 1e-2, 1e-1]
batch_sizes = [32, 64, 128]
optimizers = ['SGD', 'Adam', 'RMSprop']

hyper_results = []
for lr, bs, opt_name in product(learning_rates, batch_sizes, optimizers):
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # Model
    feat_dim = full_ds[0][0].shape[0]
    model = LogisticRegressionTorch(feat_dim, len(full_ds.classes))

    # Optimizer
    if opt_name == 'SGD': optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'Adam': optimizer = optim.Adam(model.parameters(), lr=lr)
    else: optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Тренировка 10 epoch
    for epoch in range(10):
        print(f"Training epoch {epoch}/10")
        train(model, train_loader, nn.CrossEntropyLoss(), optimizer)

    metrics, _, _ = evaluate(model, val_loader, len(full_ds.classes))
    hyper_results.append({'optimizer': opt_name, 'lr': lr, 'batch_size': bs, 'val_accuracy': metrics['precision']})

# Сохранение и визуализация
df_hyper = pd.DataFrame(hyper_results)
df_hyper.to_csv('plots/hyper_results.csv', index=False)

plt.figure(figsize=(8,6))
for opt_name in optimizers:
    subset = df_hyper[df_hyper.optimizer == opt_name]
    plt.plot(subset.lr, subset.val_accuracy, label=opt_name)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Precision')
plt.title('Hyperparameter Tuning')
plt.legend()
plt.savefig('plots/hyperparam_tuning.png')
plt.show()


### 3.2 Feature Engineering (10 баллов)
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью

# Не успел выполнить 3.2