import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.util import get_noise_matrix


def get_all_files_in_directory_and_subdirectories(dir_path: str,
                                                  extensions_list: list,
                                                  prefix_list: list) -> (list, list):
    path_list = []
    coordinate_list = []
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            extension = name.split('.')[-1]
            base_name = re.search(r'\d+_\d+_', name)[0]
            prefix = re.search(r'[A-Za-z]+', name)[0]
            if extension in extensions_list and prefix == prefix_list[0]:
                coordinate = re.findall(r'\d+', base_name)
                coordinate = list(map(int, coordinate))
                coordinate_list.append(coordinate)
                paths = []
                for prefix in prefix_list:
                    paths.append(os.path.join(path, f'{base_name}{prefix}.{extension}'))
                path_list.append(paths)
    return path_list, coordinate_list


class MarsEarthDataset(Dataset):
    def __init__(self, mars_dir: str, earth_dir: str, transform, image_size):
        self.transform = transform
        self.image_size = image_size

        self.mars_dir = mars_dir
        self.earth_dir = earth_dir

        self.mars_file_paths, self.mars_coordinates = get_all_files_in_directory_and_subdirectories(self.mars_dir,
                                                                                                    ['jpg', 'png'],
                                                                                                    ['clr', 'feature', 'topography'])
        self.earth_file_paths, self.earth_coordinates = get_all_files_in_directory_and_subdirectories(self.earth_dir,
                                                                                                      ['jpg', 'png'],
                                                                                                      ['clr', 'feature', 'topography'])

    def get_noise_layers(self, layers_count: int):
        noise_layers = np.zeros((self.image_size, self.image_size, layers_count))
        for i in range(layers_count):
            noise_layers[:,:,i] = get_noise_matrix(self.image_size)
        return self.transform(noise_layers)

    def __len__(self):
        return len(self.mars_file_paths)

    def __getitem__(self, mars_idx):
        mars_path = self.mars_file_paths[mars_idx]
        mars_coordinate = self.mars_coordinates[mars_idx]

        earth_idx = random.randrange(len(self.earth_file_paths))
        earth_path = self.earth_file_paths[earth_idx]
        earth_coordinate = self.earth_coordinates[earth_idx]

        mars_clr = self.load_img(mars_path[0])
        mars_topography = self.load_img(mars_path[2])
        mars_noise = self.get_noise_layers(3)
        mars = torch.vstack((mars_clr, mars_topography, mars_noise))

        earth_clr = self.load_img(earth_path[0])
        earth_topography = self.load_img(earth_path[2])
        earth_noise = self.get_noise_layers(3)
        earth = torch.vstack((earth_clr, earth_topography, earth_noise))

        return mars, mars_path, mars_coordinate, earth, earth_path, earth_coordinate

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


def collate_batch(batch):
    A, A_paths, A_coordinates, B, B_paths, B_coordinates = zip(*batch)
    A = torch.stack(A).type(torch.FloatTensor)
    B = torch.stack(B).type(torch.FloatTensor)
    return {'A': A, 'B': B,
            'A_paths': A_paths, 'B_paths': B_paths,
            'A_coordinates': A_coordinates, 'B_coordinates': B_coordinates}
