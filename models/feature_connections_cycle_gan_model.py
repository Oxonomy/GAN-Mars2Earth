import torch
import itertools
from util.image_pool import ImagePool
from util.util import torch_tensor_stack_images_batch_channel

from . import networks
from .cycle_gan_model import CycleGANModel


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

        self.fake_B = self.netG_A(real_A)  # G_A(A)
        fake_B = torch_tensor_stack_images_batch_channel(self.fake_B, self.real_a)
        self.rec_A = self.netG_B(fake_B)  # G_B(G_A(A))

        self.fake_A = self.netG_B(real_B)  # G_B(B)
        fake_A = torch_tensor_stack_images_batch_channel(self.fake_A, self.real_b)
        self.rec_B = self.netG_A(fake_A)  # G_A(G_B(B))
