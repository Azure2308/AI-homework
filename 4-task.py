import random
import os
import glob
from collections import defaultdict
import numpy as np
from PIL import Image
from torchvision import transforms


class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name: str, augmentation_func):
        self.augmentations[name] = augmentation_func

    def remove_augmentation(self, name: str):
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image: Image.Image) -> Image.Image:
        result = image
        for func in self.augmentations.values():
            result = func(result)
        return result

    def get_augmentations(self):
        return list(self.augmentations.keys())

# Функции-преобразования
def random_horizontal_flip(img: Image.Image, prob=0.5):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < prob else img

def random_rotation(img: Image.Image, degrees=(-10, 10)):
    angle = random.uniform(*degrees)
    return img.rotate(angle, expand=True)

def gaussian_noise(img: Image.Image, mean=0, stddev=10):
    arr = np.array(img)
    noise = np.random.normal(mean, stddev, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def color_jitter(img: Image.Image, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)

def resize_to_square(img: Image.Image, size=224):
    return img.resize((size, size))

def random_crop(img: Image.Image, crop_size=(200, 200)):
    w, h = img.size
    new_w, new_h = crop_size
    left = random.randint(0, max(0, w - new_w))
    top = random.randint(0, max(0, h - new_h))
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h))

def random_grayscale(img: Image.Image, prob=0.3):
    if random.random() < prob:
        return img.convert('L').convert('RGB')
    return img

# Создаём пайплайны
pipeline_light = AugmentationPipeline()
pipeline_medium = AugmentationPipeline()
pipeline_heavy = AugmentationPipeline()

pipeline_light.add_augmentation("resize", lambda im: resize_to_square(im))
pipeline_light.add_augmentation("flip", random_horizontal_flip)
pipeline_light.add_augmentation("jitter", lambda im: color_jitter(im, brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05))

pipeline_medium.add_augmentation("resize", lambda im: resize_to_square(im))
pipeline_medium.add_augmentation("flip", random_horizontal_flip)
pipeline_medium.add_augmentation("rotate", random_rotation)
pipeline_medium.add_augmentation("jitter", lambda im: color_jitter(im, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

pipeline_heavy.add_augmentation("resize", lambda im: resize_to_square(im))
pipeline_heavy.add_augmentation("flip", random_horizontal_flip)
pipeline_heavy.add_augmentation("rotate", lambda im: random_rotation(im, degrees=(-30, 30)))
pipeline_heavy.add_augmentation("noise", gaussian_noise)
pipeline_heavy.add_augmentation("jitter", lambda im: color_jitter(im, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
pipeline_heavy.add_augmentation("crop", random_crop)
pipeline_heavy.add_augmentation("grayscale", random_grayscale)

images_dir = 'data/train'
output_dirs = ['results/light', 'results/medium', 'results/heavy']
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

saved_counts = defaultdict(int)
max_per_pipeline = 5

# Собираем все изображения
extensions = ['*.jpg', '*.jpeg', '*.png']
all_images = []
for ext in extensions:
    all_images.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))

# Применяем и сохраняем
for img_path in all_images:
    img = Image.open(img_path)
    fname = os.path.basename(img_path)

    if saved_counts['light'] < max_per_pipeline:
        out = pipeline_light.apply(img)
        out.save(os.path.join(output_dirs[0], fname))
        saved_counts['light'] += 1

    if saved_counts['medium'] < max_per_pipeline:
        out = pipeline_medium.apply(img)
        out.save(os.path.join(output_dirs[1], fname))
        saved_counts['medium'] += 1

    if saved_counts['heavy'] < max_per_pipeline:
        out = pipeline_heavy.apply(img)
        out.save(os.path.join(output_dirs[2], fname))
        saved_counts['heavy'] += 1

    if all(saved_counts[mode] >= max_per_pipeline for mode in ['light', 'medium', 'heavy']):
        break

print("Выполнено")