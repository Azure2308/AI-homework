import os
from turtledemo.penrose import start

from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from utils.extra_augs import *

class Augmenter:
    # Применяет размытие по Гауссу с заданной вероятностью
    def gaussian_blur_randomly(self, image, radius_limit=4, prob=0.5):
        if random.random() < prob:
            radius = random.uniform(0, radius_limit)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    # Перспективное искажение с варьирующимся сдвигом
    def perspective_warp(self, image, scale=0.4, prob=0.5):
        if random.random() >= prob:
            return image

        w, h = image.size
        origin_pts = [(0, 0), (w, 0), (w, h), (0, h)]
        dx = int(scale * w)
        dy = int(scale * h)

        dest_pts = [
            (
                x + random.uniform(-dx, dx),
                y + random.uniform(-dy, dy)
            )
            for (x, y) in origin_pts
        ]

        return TF.perspective(image, origin_pts, dest_pts)

    # Изменение яркости и контрастности
    def tweak_brightness_contrast(self, image, bright_range=(0.6, 1.4), contrast_range=(0.6, 1.4), prob=0.5):
        if random.random() >= prob:
            return image

        brightness = random.uniform(*bright_range)
        contrast = random.uniform(*contrast_range)

        img = ImageEnhance.Brightness(image).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        return img


base_path = 'data/train'
augmentor = Augmenter()
selected_classes = sorted(os.listdir(base_path))[:5]
sample_paths = [os.path.join(base_path, cls, os.listdir(os.path.join(base_path, cls))[0]) for cls in selected_classes]

# Визуализация изображения
def visualize(ax, image, title=None):
    tensor_img = T.ToTensor()(image) if not isinstance(image, torch.Tensor) else image
    img_array = tensor_img.permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_array, 0, 1))
    ax.set_title(title or "", fontsize=10)
    ax.axis('off')


for i, path in enumerate(sample_paths, start=1):
    image = Image.open(path).convert("RGB")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))

    # Новые аугментации
    blurred = augmentor.gaussian_blur_randomly(image, radius_limit=8, prob=1.0)
    warped = augmentor.perspective_warp(image, scale=0.3, prob=1.0)
    enhanced = augmentor.tweak_brightness_contrast(image, bright_range=(0.7, 1.3), contrast_range=(0.7, 1.3), prob=1.0)

    visualize(axes[0][0], blurred, 'Gaussian Blur')
    visualize(axes[0][1], warped, 'Perspective Warp')
    visualize(axes[0][2], enhanced, 'Brightness & Contrast')

    # Аугментации из extra_augs (с пары)
    noise = T.Compose([ T.ToTensor(), AddGaussianNoise(0., 0.2) ])
    eraser = T.Compose([ T.ToTensor(), RandomErasingCustom(p=1.0) ])
    posterizer = T.Compose([ T.ToTensor(), Posterize(bits=4) ])

    noisy_img = noise(image)
    erased_img = eraser(image)
    posterized_img = posterizer(image)

    visualize(axes[1][0], noisy_img, 'Gaussian Noise')
    visualize(axes[1][1], erased_img, 'Erasing')
    visualize(axes[1][2], posterized_img, 'Posterize')

    plt.tight_layout()
    plt.savefig(f'plots/2-task-{i}')
    plt.show()
