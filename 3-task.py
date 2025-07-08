import os
import glob
from pathlib import Path
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_path = Path('data')


# Собираем информацию по изображениям: размер, набор, метка класса
def gather_image_stats(root: Path, file_exts=None):
    if file_exts is None:
        file_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    records = []
    splits = ['train', 'test']

    for split in splits:
        split_dir = root / split
        for cls_name in sorted(split_dir.iterdir()):
            if not cls_name.is_dir():
                continue

            path_pattern = [str(cls_name / ext) for ext in file_exts]
            image_files = []
            for pattern in path_pattern:
                image_files.extend(glob.glob(pattern))

            print(f"В {split}/{cls_name.name} найдено {len(image_files)} файлов")

            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        records.append({
                            'split': split,
                            'label': cls_name.name,
                            'width': w,
                            'height': h
                        })
                except Exception as err:
                    print(f"Ошибка в {img_path}, error: {err}")

    return pd.DataFrame.from_records(records)


# Визуализация распределения размеров
def plot_size_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    for split, color in zip(['train', 'test'], ['skyblue', 'salmon']):
        subset = df[df['split'] == split]
        sns.histplot(subset['height'], bins=25, kde=True, alpha=0.4, label=f"{split} Height", color=color)
        sns.histplot(subset['width'], bins=25, kde=True, alpha=0.4, label=f"{split} Width", color=color)

    plt.title("Распределение размеров")
    plt.xlabel("Пиксели")
    plt.ylabel("Количество изображений")
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/3-task-graph-1.png')
    plt.show()


# Визуализация количества изображений по классам
def plot_class_counts(df: pd.DataFrame):
    count_df = df.groupby(['split', 'label']).size().reset_index(name='count')
    g = sns.catplot(
        data=count_df,
        x='label', y='count', hue='split',
        kind='bar', height=6, aspect=1.5
    )
    g.set_axis_labels("Класс", "Кол-во изображений")
    g.fig.suptitle("Кол-во изображений в классе")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/3-task-graph-2.png')
    plt.show()


if __name__ == '__main__':
    stats_df = gather_image_stats(root_path)

    print(f"Средний размер: height={stats_df['height'].mean():.1f}, width={stats_df['width'].mean():.1f}")
    print(f"Минимальный размер: {stats_df[['height', 'width']].min().to_dict()}")
    print(f"максимальный размер: {stats_df[['height', 'width']].max().to_dict()}")

    plot_size_distribution(stats_df)
    plot_class_counts(stats_df)