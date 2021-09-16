import os
import time

import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

import config as c
from dataset import MarsEarthDataset
from models.cycle_gan_model import CycleGANModel
from models.feature_connections_cycle_gan_model import FeatureConnectionsCycleGANModel
from util.visualizer import Visualizer


def collate_batch(batch):
    A, B, A_paths, B_paths = zip(*batch)
    A = torch.stack(A)
    B = torch.stack(B)
    return {'A': A, 'B': B, 'A_paths': A_paths, 'B_paths': B_paths}


def display_current_results(model):
    model.compute_visuals()
    images = model.get_current_visuals()
    wandb_images = []
    for image_name in images.keys():
        image = images[image_name]
        image = (image.cpu().detach().numpy() + 1)[0] / 2.0 * 255.0
        image = np.transpose(image, (1, 2, 0))
        wandb_images.append(wandb.Image(image, caption=image_name))
    wandb.log({"img": wandb_images})


def train(model, data_loader):
    total_iters = 0
    for epoch in range(c.EPOCH_COUNT, c.N_EPOCHS + c.EPOCH_COUNT):
        epoch_iter = 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        for data in tqdm(data_loader):

            total_iters += c.BATCH_SIZE
            epoch_iter += c.BATCH_SIZE
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % 10 == 0:
                display_current_results(model)

            wandb.log(model.get_current_losses())

            if total_iters % 50000 == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters
                model.save_networks(save_suffix)

        if epoch % 50 == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


def build_data_loader():
    dataset = MarsEarthDataset(earth_dir=os.path.join(c.DATA_ROOT, 'earth'),
                               mars_dir=os.path.join(c.DATA_ROOT, 'mars'),
                               image_size=c.IMAGE_SIZE,
                               transform=transforms.Compose([
                                   transforms.Resize(c.IMAGE_SIZE),
                                   transforms.CenterCrop(c.IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
                               )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=c.BATCH_SIZE,
                                              shuffle=True, collate_fn=collate_batch
                                              )
    return dataset, data_loader


if __name__ == '__main__':
    wandb.init(project=c.NAME)

    dataset, data_loader = build_data_loader()

    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = FeatureConnectionsCycleGANModel(name=c.NAME, netG='resnet_9blocks')
    model.setup(continue_train=False, epoch_count=0)
    train(model=model, data_loader=data_loader)
