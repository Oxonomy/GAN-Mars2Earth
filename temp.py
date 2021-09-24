import torch
from PIL import Image
from tqdm import tqdm
import cv2
import config as c
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from train import build_data_loader
from models.feature_connections_cycle_gan_model import FeatureConnectionsCycleGANModel
from util.util import torch_tensor_stack_images_batch_channel

img = cv2.imread('C:\\Users\\Kirill\\Documents\\DataSets\\mars2earth\\earth_clr.tif')

plt.imshow(img[5000:9000, 5000:9000])
plt.show()

data = img.reshape(img.shape[0]*img.shape[1], 3)

x = data[:1000000, 0]
y = data[:1000000, 1]
z = data[:1000000, 2]



fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()