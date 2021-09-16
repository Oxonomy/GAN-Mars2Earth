import os
import urllib.request
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_img(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def build_mars_dataset(dataset_dir, output_size, output_dir_name):
    for tile_x in 'ABCD':
        for tile_y in [1, 2]:
            url = f'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74117/world.200408.3x21600x21600.{tile_x}{tile_y}.jpg'
            img = load_img(url)

            height, width, channels = img.shape
            for x in tqdm(range(0, width - output_size - 1, output_size)):
                for y in range(0, height - output_size - 1, output_size):
                    path = os.path.join(dataset_dir, f"{output_dir_name}\\{tile_x}{tile_y}_{x}_{y}.png")
                    cut_img = img[y:y + output_size, x:x + output_size]

                    if np.mean(cut_img) > 12:
                        cv2.imwrite(path, cut_img)


if __name__ == '__main__':
    dataset_dir = 'C:\\Users\\Kirill\\Documents\\DataSets\\earth'
    output_size = 128
    build_mars_dataset(dataset_dir, output_size, f'data_{output_size}_drain_ocean')
