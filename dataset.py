import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


def get_all_files_in_directory_and_subdirectories(dir_path: str,
                                                  extensions_list: list) -> list:
    path_list = []
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.split('.')[-1] in extensions_list:
                path_list.append(os.path.join(path, name))
    return path_list


class MarsEarthDataset(Dataset):
    def __init__(self, mars_dir: str, earth_dir: str, transform, image_size):
        self.transform = transform
        self.image_size = image_size

        self.mars_dir = mars_dir
        self.earth_dir = earth_dir

        self.mars_file_paths = get_all_files_in_directory_and_subdirectories(self.mars_dir,
                                                                        ['jpg', 'png'])
        self.earth_file_paths = get_all_files_in_directory_and_subdirectories(self.earth_dir,
                                                                         ['jpg', 'png'])

    def __len__(self):
        return len(self.mars_file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mars_path = self.mars_file_paths[idx]
        earth_path = random.choice(self.earth_file_paths)

        mars_img = self.load_img(mars_path)
        earth_img = self.load_img(earth_path)

        return mars_img, earth_img, mars_path, earth_path

    def load_img(self, path):
        img = Image.open(path)
        img = self.transform(img)
        return img

    def augmentations(self):
        pass

    def load_mars_map(self, x_0, y_0, x_1, y_1):
        x_0 = x_0 // self.image_size * self.image_size
        y_0 = y_0 // self.image_size * self.image_size
        x_1 = x_1 // self.image_size * self.image_size
        y_1 = y_1 // self.image_size * self.image_size

        images = []
        for x in range(x_0, x_1, self.image_size):
            images.append([])
            for y in range(y_0, y_1, self.image_size):
                path = os.path.join(self.mars_dir, f'{x}_{y}.png')
                images[-1].append(self.load_img(path))
            images[-1] = torch.stack(images[-1])
        return images
