import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from homework_datasets import MushroomDataset


### 1.2 Расширение логистической регрессии (15 баллов)
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix

class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, num_classes):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb)
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_true.append(yb.cpu().numpy())
            all_pred.append(preds)
            all_prob.append(probs)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')

    metrics = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall':    recall_score(y_true, y_pred, average='macro'),
        'f1':        f1_score(y_true, y_pred, average='macro'),
        'roc_auc':   roc_auc
    }
    return metrics, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel='True', xlabel='Predicted',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Константы
    DATA_PATH = './data/mushrooms.csv'
    BATCH_SIZE = 64
    LR = 1e-2
    EPOCHS = 50
    TEST_RATIO = 0.2
    VAL_RATIO = 0.1

    # Разбивка датасета
    ds = MushroomDataset(DATA_PATH, target_col='class')
    num_classes = len(ds.classes)
    n = len(ds)
    n_test = int(TEST_RATIO * n)
    n_val = int(VAL_RATIO * (n - n_test))
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Инициализация модели, критерия и оптимизатора
    feat_dim = ds[0][0].shape[0]
    model = LogisticRegressionTorch(feat_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Тренировка
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {loss:.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), "models/logreg_mushroom.pth")
    print(f"Модель сохранена в models/logreg_mushroom.pth")

    # Оценка
    metrics, y_true, y_pred = evaluate(model, test_loader, num_classes)
    print("\nТестовые метрики:")
    for k, v in metrics.items():
        print(f"{k.capitalize():7s}: {v:.4f}")

    # Сохранение матрицы
    cm_path = "plots/mushroom_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, classes=ds.classes, save_path=cm_path)
    print(f"Матрица сохранена в {cm_path}")