import os
import numpy as np

from cv2 import cv2


def build_mars_dataset(dataset_dir, img, output_size, output_dir_name):
    height, width, channels = img.shape
    for x in range(0, width - output_size-1, output_size):
        for y in range(0, height - output_size-1, output_size):
            path = os.path.join(dataset_dir, f"{output_dir_name}\\{x}_{y}.png")
            print(x, x + output_size, y, y + output_size)
            cv2.imwrite(path, img[y:y + output_size, x:x + output_size])


if __name__ == '__main__':
    dataset_dir = 'C:\\Users\\Kirill\\Documents\\DataSets\\mars'
    output_size = 128

    img = cv2.imread(os.path.join(dataset_dir, "Mars_Viking_MDIM21_ClrMosaic_global_463m.tif"), cv2.IMREAD_COLOR)
    build_mars_dataset(dataset_dir=dataset_dir, img=img, output_size=output_size, output_dir_name='data_clr_128')
