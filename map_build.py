import torch
from tqdm import tqdm
import cv2
import config as c
import numpy as np
import matplotlib.pyplot as plt
from train import build_data_loader
from models.feature_connections_cycle_gan_model import FeatureConnectionsCycleGANModel
from util.util import torch_tensor_stack_images_batch_channel


def batch2images(batch: torch.Tensor):
    batch = batch.cpu().detach().numpy()
    images = []
    for image in batch:
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (c.IMAGE_SIZE // 2, c.IMAGE_SIZE // 2))
        images.append(image)
    return images


def image_from_array(image_list):
    image = [[]]
    current_lat = image_list[0][1][0]
    for i in range(len(image_list)):
        print(image_list[i][1])
        if current_lat != image_list[i][1][0]:
            current_lat = image_list[i][1][0]
            image[-1] = np.concatenate(image[-1], axis=0)
            image.append([])
        image[-1].append(image_list[i][0])
    image = np.concatenate(image[:-1], axis=1)
    return image


def map_build(data_loader, model):
    fake_A_list = []
    fake_B_list = []
    A_coordinates = []
    B_coordinates = []
    for data in tqdm(data_loader):
        real_A = data['A']
        real_a = data['a']
        real_B = data['B']
        real_b = data['b']

        #plt.imshow(np.transpose(real_A.cpu().detach().numpy()[0], (1, 2, 0))[:, :, :3])
        #plt.show()
        #real_a[:,:,:,2]= 0
        real_A = torch_tensor_stack_images_batch_channel(real_A, real_a)
        fake_B = model.netG_A(real_A)

        real_B = torch_tensor_stack_images_batch_channel(real_B, real_b)
        fake_A = model.netG_B(real_B)

        fake_A_list += batch2images(fake_A)
        fake_B_list += batch2images(fake_B)

        A_coordinates += data['A_coordinates']
        B_coordinates += data['B_coordinates']
        if len(fake_A_list) > 100:
            break

    fake_A_list = list(map(lambda i: (fake_A_list[i], B_coordinates[i]), range(len(fake_A_list))))
    fake_B_list = list(map(lambda i: (fake_B_list[i], A_coordinates[i]), range(len(fake_B_list))))
    fake_A_list = sorted(fake_A_list, key=lambda x: tuple(x[1]))
    fake_B_list = sorted(fake_B_list, key=lambda x: tuple(x[1]))

    fake_A = image_from_array(fake_B_list)
    fake_B = image_from_array(fake_B_list)

    cv2.imwrite('fake_A.jpg', fake_A[:, :, :3])
    cv2.imwrite('fake_B.jpg', fake_B[:, :, :3])
    plt.imshow(fake_A[:, :, :3])
    plt.show()
    # for coordinate, image in zip(A_coordinates, fake_B_list):


if __name__ == '__main__':
    dataset, data_loader = build_data_loader(shuffle=False, dry_images_percentage=1)
    model = FeatureConnectionsCycleGANModel(name=c.NAME, netG='resnet_6blocks', input_nc=9, output_nc=6, lambda_identity=0)
    model.setup(continue_train=True, epoch_count=0)
    map_build(data_loader, model)
