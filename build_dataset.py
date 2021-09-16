import os
import urllib.request
import numpy as np
from cv2 import cv2
from tqdm import tqdm


def load_img(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def load_earth_maps(dataset_dir):
    img_clr = []
    img_topography = []
    for tile_x in tqdm('ABCD'):
        img_clr.append([])
        img_topography.append([])
        for tile_y in tqdm([1, 2]):
            url_clr = f'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74117/world.200408.3x21600x21600.{tile_x}{tile_y}.jpg'
            url_topography = f'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_{tile_x}{tile_y}_grey_geo.tif'
            img_clr[-1].append(load_img(url_clr))
            img_topography[-1].append(load_img(url_topography))
    for i in range(len(img_clr)):
        img_clr[i] = np.concatenate(img_clr[i], axis=0)
        img_topography[i] = np.concatenate(img_topography[i], axis=0)
    img_topography = np.concatenate(img_topography, axis=1)
    img_clr = np.concatenate(img_clr, axis=1)
    cv2.imwrite(os.path.join(dataset_dir, 'earth_clr.tif'), img_clr)
    cv2.imwrite(os.path.join(dataset_dir, 'earth_topography.tif'), img_topography)


def build_gradient_map(img):
    grad_x = img - np.roll(img, axis=0, shift=1) + 128
    grad_y = img - np.roll(img, axis=1, shift=1) + 128
    gradient_map = np.stack([img, grad_x, grad_y], axis=0)
    gradient_map = np.swapaxes(gradient_map, 0, 2)
    gradient_map = np.swapaxes(gradient_map, 0, 1)
    return gradient_map


def save_img_cut(img, x, y, output_size, path):
    img = img[y:y + output_size, x:x + output_size]
    cv2.imwrite(path, img)


def build_feature_map(img_topography, height, width, output_size):
    img_mask = (img_topography > 0).astype(np.uint8) * 255
    img_lat = np.concatenate([np.tile(np.linspace(0, 255, num=height // 2, dtype=np.uint8), (width, 1)).T,
                              np.tile(np.linspace(255, 0, num=height // 2, dtype=np.uint8), (width, 1)).T])

    img_alt = cv2.resize(img_topography, dsize=(width // output_size, height // output_size), interpolation=cv2.INTER_CUBIC)
    img_alt = cv2.resize(img_alt, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    feature_map = np.stack([img_mask, img_lat, img_alt], axis=0)
    feature_map = np.swapaxes(feature_map, 0, 2)
    feature_map = np.swapaxes(feature_map, 0, 1)
    return feature_map


def build_dataset(dataset_dir, planet, output_size=128):
    img_clr = cv2.imread(os.path.join(dataset_dir, f'{planet}_clr.tif'))
    img_topography = cv2.imread(os.path.join(dataset_dir, f'{planet}_topography.tif'), -1)

    height, width, channels = img_clr.shape

    img_feature = build_feature_map(img_topography, height, width, output_size)
    img_topography = build_gradient_map(img_topography)

    height, width, channels = img_clr.shape
    for x in tqdm(range(0, width - output_size - 1, output_size)):
        for y in range(0, height - output_size - 1, output_size):
            path_topography = os.path.join(dataset_dir, f"{planet}_{output_size}\\{x}_{y}_topography.png")
            path_clr = os.path.join(dataset_dir, f"{planet}_{output_size}\\{x}_{y}_clr.png")
            path_feature = os.path.join(dataset_dir, f"{planet}_{output_size}\\{x}_{y}_feature.png")

            save_img_cut(img_topography, x, y, output_size, path_topography)
            save_img_cut(img_clr, x, y, output_size, path_clr)
            save_img_cut(img_feature, x, y, output_size, path_feature)


if __name__ == '__main__':
    build_dataset(dataset_dir='C:\\Users\\Kirill\\Documents\\DataSets\\mars2earth', planet='earth', output_size=256)
    build_dataset(dataset_dir='C:\\Users\\Kirill\\Documents\\DataSets\\mars2earth', planet='mars', output_size=256)
