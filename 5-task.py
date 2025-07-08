import timeit
import pandas as pd
import psutil
import gc
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms


def benchmark_timer(target_func, parameters=(), runs=10):
    timer = timeit.Timer(lambda: target_func(*parameters))
    timings = timer.repeat(number=runs)
    return sum(timings) / len(timings)


def current_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def preprocess_image(path, img_size):
    aug_pipeline = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor_image = aug_pipeline(image)
    return tensor_image


def log(message):
    print(f"[LOG] {message}")


img_root = 'data/train/'
image_files = glob.glob(os.path.join(img_root, '*/*'))

# Размеры для тестирования
sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]

metrics = {"resolution": [], "avg_time": [], "memory_delta": []}

for size in sizes:
    res_str = f"{size[0]}x{size[1]}"
    log(f"Обработка изображений размера {res_str}")

    total_time = 0
    start_memory = current_memory_usage()

    for idx, img_file in enumerate(image_files[:100]):
        if idx % 10 == 0:
            log(f"Обрабатывается изображение {idx + 1}/100")

        img_time = benchmark_timer(preprocess_image, parameters=(img_file, size))
        total_time += img_time
        gc.collect()

    metrics["resolution"].append(size[0])
    metrics["avg_time"].append(total_time / 100)
    metrics["memory_delta"].append(current_memory_usage() - start_memory)

log("Выполнено. Формируются графики")

results_df = pd.DataFrame(metrics)

plt.figure(figsize=(10, 6))
sns.lineplot(x="resolution", y="avg_time", marker="o", data=results_df, color='blue')
plt.title("Среднее время загрузки и аугментации изображений")
plt.xlabel("Размер изображения (пиксели)")
plt.ylabel("Время (сек)")
plt.grid(True)
plt.savefig('plots/5-task-graph-1.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x="resolution", y="memory_delta", marker="o", data=results_df, color='green')
plt.title("Изменение потребления памяти при обработке изображений")
plt.xlabel("Размер изображения (пиксели)")
plt.ylabel("Память (МБ)")
plt.grid(True)
plt.savefig('plots/5-task-graph-2.png')
plt.show()