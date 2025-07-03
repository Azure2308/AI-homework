import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from homework_datasets import PokemonDataset

### 1.1 Расширение линейной регрессии (15 баллов)
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping

class LinearRegressionTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, X: torch.Tensor):
        return self.linear(X)

# Константы
DATA_PATH = './data/Pokemon.csv'
TARGET = 'Total'
LEARNING_RATE = 1e-3
EPOCHS = 100
L1_LAMBDA = 1e-4
L2_LAMBDA = 1e-3
PATIENCE = 10
BATCH_SIZE = 32
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Тренировочная функция с early stopping
def train_with_early_stopping(model,
                               train_loader,
                               val_loader,
                               criterion,
                               optimizer,
                               l1_lambda,
                               l2_lambda,
                               patience,
                               epochs):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_mse = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            mse_loss = criterion(preds, y_batch)
            # Регуляризация
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = mse_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            total_mse += mse_loss.item()
        avg_train_mse = total_mse / len(train_loader)

        # Валидация
        model.eval()
        total_val_mse = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                total_val_mse += criterion(model(X_val), y_val).item()
        avg_val_mse = total_val_mse / len(val_loader)

        print(f"[Epoch {epoch}] Тренировка MSE: {avg_train_mse:.4f} | Валидация MSE: {avg_val_mse:.4f}")

        # Early Stopping
        if avg_val_mse < best_val_loss - 1e-4:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/pokemon_linreg.pth")
            print("Обнаружено улучшение, сохранено.")
        else:
            epochs_no_improve += 1
            print(f"Нет улучшений для {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(" Ранняя остановка.")
                break

if __name__ == '__main__':
    # Загрузка и разбивка датасета
    dataset = PokemonDataset(DATA_PATH, target=TARGET)
    n = len(dataset)
    n_test = int(TEST_RATIO * n)
    n_val = int(VAL_RATIO * (n - n_test))
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Размеры -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Инициализация модели, критерия и оптимизатора
    feature_dim = dataset[0][0].shape[0]
    model = LinearRegressionTorch(in_features=feature_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Тренировка
    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        l1_lambda=L1_LAMBDA,
        l2_lambda=L2_LAMBDA,
        patience=PATIENCE,
        epochs=EPOCHS
    )

    # Оценка
    model.load_state_dict(torch.load("models/pokemon_linreg.pth"))
    model.eval()
    total_test_mse = 0.0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            total_test_mse += criterion(model(X_test), y_test).item()
    avg_test_mse = total_test_mse / len(test_loader)
    print(f"Финальный тест MSE: {avg_test_mse:.4f}")