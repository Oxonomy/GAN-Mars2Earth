import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_all_files_in_directory_and_subdirectories(dir_path: str,
                                                  extensions_list: list,
                                                  prefix_list: list) -> list:
    path_list = []
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            extension = name.split('.')[-1]
            base_name = re.search(r'\d+_\d+_', name)[0]
            prefix = re.search(r'[A-Za-z]+', name)[0]
            if extension in extensions_list and prefix == prefix_list[0]:
                paths = []
                for prefix in prefix_list:
                    paths.append(os.path.join(path, f'{base_name}{prefix}.{extension}'))
                path_list.append(paths)
    return path_list


class MarsEarthDataset(Dataset):
    def __init__(self, mars_dir: str, earth_dir: str, transform, image_size, dry_images_percentage=0.01):
        self.transform = transform
        self.image_size = image_size

        self.mars_dir = mars_dir
        self.earth_dir = earth_dir

        self.mars_file_paths = get_all_files_in_directory_and_subdirectories(self.mars_dir,
                                                                             ['jpg', 'png'],
                                                                             ['clr', 'feature', 'topography'])
        self.earth_file_paths = get_all_files_in_directory_and_subdirectories(self.earth_dir,
                                                                              ['jpg', 'png'],
                                                                              ['clr', 'feature', 'topography'])
        if dry_images_percentage < 1:
            earth_dry_mask = self.get_dry_mask(self.earth_file_paths, dry_images_percentage)
            self.earth_file_paths = list(np.array(self.earth_file_paths)[earth_dry_mask])


    def __len__(self):
        return len(self.mars_file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mars_path = self.mars_file_paths[idx]
        earth_path = random.choice(self.earth_file_paths)

        mars_clr = self.load_img(mars_path[0])
        mars_feature = self.load_img(mars_path[1])
        mars_topography = self.load_img(mars_path[2])
        mars = torch.vstack((mars_clr, mars_topography))

        earth_clr = self.load_img(earth_path[0])
        earth_feature = self.load_img(earth_path[1])
        earth_topography = self.load_img(earth_path[2])
        earth = torch.vstack((earth_clr, earth_topography))

        return mars, mars_feature, mars_path, earth, earth_feature, earth_path

    def load_img(self, path):
        img = Image.open(path)
        img = self.transform(img)
        return img

    def augmentations(self):
        pass

    def get_dry_mask(self, paths, dry_images_percentage=0.01) -> np.ndarray:
        mask = np.zeros(len(paths), dtype=bool)
        for i, path in enumerate(paths):
            feature_map = Image.open(path[1])
            feature_map = np.asarray(feature_map)
            if np.max(feature_map[:, :, 0]) > 0 or dry_images_percentage > random.random():
                mask[i] = True
        return mask

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
