import os
import time
import cv2
import PIL
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

import config as c
from dataset import MarsEarthDataset
from models.cycle_gan_model import CycleGANModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""
c.BATCH_SIZE = 1

dataset = MarsEarthDataset(earth_dir=os.path.join(c.DATA_ROOT, 'earth\\data_256_drain_ocean'),
                           mars_dir=os.path.join(c.DATA_ROOT, 'mars\\data_clr_256'),
                           image_size=c.IMAGE_SIZE,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                           )
PIL.Image.MAX_IMAGE_PIXELS = 9331200000

model = CycleGANModel(name=c.NAME, gpu_ids=[])
model.setup(continue_train=True)

img = dataset.load_img(os.path.join(c.DATA_ROOT, 'mars\\Mars_Viking_MDIM21_ClrMosaic_global_3704m.jpg'))

image = model.netG_A(torch.stack([img]))
image = (image.cpu().detach().numpy() + 1)[0] / 2.0 * 255.0
image = np.transpose(image, (1, 2, 0))

cv2.imwrite("filename.png", image)
