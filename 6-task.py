import torch
import torchvision.models as vision_models
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.datasets import CustomImageDataset
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Аугментация
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_data = CustomImageDataset('data/train/', transform=image_transform)
val_data = CustomImageDataset('data/test/', transform=image_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

net = vision_models.resnet18(weights=vision_models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

total_classes = len(train_data.get_class_names())
net.fc = torch.nn.Linear(net.fc.in_features, total_classes).to(DEVICE)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
loss_function = torch.nn.CrossEntropyLoss()

metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

EPOCHS = 15
# Обучение
for epoch in range(EPOCHS):
    net.train()
    epoch_loss = 0.0
    correct_train = 0
    total_train = 0

    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        preds = net(imgs)
        loss = loss_function(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, pred_classes = torch.max(preds.data, 1)
        total_train += targets.size(0)
        correct_train += (pred_classes == targets).sum().item()

    avg_train_loss = epoch_loss / len(train_loader)
    acc_train = correct_train / total_train
    metrics['train_loss'].append(avg_train_loss)
    metrics['train_acc'].append(acc_train)

    # Валидация
    net.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = net(imgs)
            val_loss = loss_function(preds, targets)
            val_loss_total += val_loss.item()

            _, pred_classes = torch.max(preds.data, 1)
            val_total += targets.size(0)
            val_correct += (pred_classes == targets).sum().item()

    avg_val_loss = val_loss_total / len(val_loader)
    acc_val = val_correct / val_total
    metrics['val_loss'].append(avg_val_loss)
    metrics['val_acc'].append(acc_val)

    print(f"Epochs: {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {acc_train:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc_val:.4f}")

# Визуализация всех метрик на одном графике
plt.figure(figsize=(10, 6))

plt.plot(metrics['train_loss'], label='Train Loss', linestyle='--', marker='o')
plt.plot(metrics['val_loss'], label='Validation Loss', linestyle='--', marker='x')
plt.plot(metrics['train_acc'], label='Train Accuracy', linestyle='-', marker='o')
plt.plot(metrics['val_acc'], label='Validation Accuracy', linestyle='-', marker='x')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training & Validation Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/6-task-graph.png')
plt.show()