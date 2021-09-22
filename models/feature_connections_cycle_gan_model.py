import torch
from util.image_pool import ImagePool
from util.util import torch_tensor_stack_images_batch_channel

from . import networks
from .cycle_gan_model import CycleGANModel


def recover_image_from_tiles(tiles: list, tile_axis_count: int = 2):
    image = []
    for i in range(0, len(tiles), tile_axis_count):
        image.append(torch.cat(tiles[i:i + tile_axis_count], 3))
    image = torch.cat(image, 2)
    return image


def cut_tensor_into_tiles(batch: torch.tensor, tile_axis_count: int = 2) -> list:
    weight, high = batch.shape[2:]
    assert high % tile_axis_count == 0 and weight % tile_axis_count == 0
    tiles = []
    for i in range(tile_axis_count):
        for j in range(tile_axis_count):
            tiles.append(batch[:, :,
                         weight // tile_axis_count * i: weight // tile_axis_count * (i + 1),
                         high // tile_axis_count * j: high // tile_axis_count * (j + 1),
                         ])
    return tiles


def tiles_netG_prediction(netG, input: torch.tensor):
    tiles = cut_tensor_into_tiles(input)
    for i, tile in enumerate(tiles):
        tiles[i] = netG(tile)
    return recover_image_from_tiles(tiles)


class FeatureConnectionsCycleGANModel(CycleGANModel):

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_a = input['a' if AtoB else 'b'].to(self.device)

        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_b = input['b' if AtoB else 'a'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_A = torch_tensor_stack_images_batch_channel(self.real_A, self.real_b)
        real_B = torch_tensor_stack_images_batch_channel(self.real_B, self.real_a)

        self.fake_B = tiles_netG_prediction(self.netG_A, real_A)  # G_A(A)
        fake_B = torch_tensor_stack_images_batch_channel(self.fake_B, self.real_a)
        self.rec_A = tiles_netG_prediction(self.netG_B, fake_B)  # G_B(G_A(A))

        self.fake_A = tiles_netG_prediction(self.netG_B, real_B)  # G_B(B)
        fake_A = torch_tensor_stack_images_batch_channel(self.fake_A, self.real_b)
        self.rec_B = tiles_netG_prediction(self.netG_A, fake_A)  # G_A(G_B(B))

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = torch_tensor_stack_images_batch_channel(self.fake_B, self.real_b)
        fake_B = self.fake_B_pool.query(fake_B)

        real_B = torch_tensor_stack_images_batch_channel(self.real_B, self.real_b)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = torch_tensor_stack_images_batch_channel(self.fake_A, self.real_a)
        fake_A = self.fake_A_pool.query(fake_A)

        real_A = torch_tensor_stack_images_batch_channel(self.real_A, self.real_a)
        self.loss_D_B = self.backward_D_basic(self.netD_B, real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        self.loss_idt_A = 0
        self.loss_idt_B = 0

        fake_B = torch_tensor_stack_images_batch_channel(self.fake_B, self.real_b)
        fake_A = torch_tensor_stack_images_batch_channel(self.fake_A, self.real_a)
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

