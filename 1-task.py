import os
from turtledemo.penrose import start

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

root = 'data/train'

# Словарь стандартных аугментаций
standart_aug = {
    'Original': None,
    'Horizontal Flip': transforms.RandomHorizontalFlip(p=1),
    'Random Crop': transforms.RandomCrop(size=(200, 200)),
    'Color Jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    'Rotation': transforms.RandomRotation(degrees=(-45, 45)),
    'Grayscale': transforms.RandomGrayscale(p=1),
}


aug_names = list(standart_aug.keys()) + ['Всё вместе']

class_folders = sorted(os.listdir(root))[:5]
image_paths = []
for folder in class_folders:
    images_in_folder = os.listdir(os.path.join(root, folder))
    first_image = os.path.join(root, folder, images_in_folder[0])
    image_paths.append(first_image)

for j, img_path in enumerate(image_paths, start=1):
    original_img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(nrows=1, ncols=len(aug_names), figsize=(20, 5))

    for i, name in enumerate(aug_names):
        if name == 'Original':
            img_to_show = original_img.copy()
        elif name == 'Всё вместе':
            transformed_img = original_img
            for t in standart_aug.values():
                if t is not None:
                    transformed_img = t(transformed_img)
            img_to_show = transformed_img
        else:
            transform = standart_aug[name]
            img_to_show = transform(original_img)

        axes[i].imshow(img_to_show)
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.savefig(f'plots/1-task-{j}')
    plt.show()