import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, lambda_A: float = 10.0, lambda_B: float = 10.0, lambda_identity: float = 0.5,
                 input_nc: int = 3, output_nc: int = 3, ngf: int = 64, ndf: int = 64,
                 netD: str = 'basic', netG: str = 'resnet_9blocks',
                 n_layers_D: int = 3,
                 norm: str ='instance',
                 init_type: str = 'normal',
                 init_gain: float = 0.02,
                 no_dropout: bool = True,
                 direction: str = 'AtoB',

                 beta1: float = 0.5,
                 lr: float = 0.0002,
                 gan_mode: str = 'lsgan',
                 pool_size: int = 50,

                 gpu_ids=[0],
                 checkpoints_dir = './checkpoints',
                 name = 'experiment_name',
                 preprocess = 'resize_and_crop',
                 lr_policy = 'linear',
                 lr_decay_iters = 50):
        """
        
        :param lambda_A: weight for cycle loss (A -> B -> A)
        :param lambda_B: weight for cycle loss (B -> A -> B)
        :param lambda_identity: use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.
         For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1
        :param input_nc: # of input image channels: 3 for RGB and 1 for grayscale
        :param output_nc: # of output image channels: 3 for RGB and 1 for grayscale
        :param ngf: # of gen filters in the last conv layer
        :param ndf: # of discrim filters in the first conv layer
        :param netD: specify discriminator architecture [basic | n_layers | pixel].
         The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
        :param netG: specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        :param n_layers_D: only used if netD==n_layers
        :param norm: instance normalization or batch normalization [instance | batch | none]
        :param init_type: network initialization [normal | xavier | kaiming | orthogonal]
        :param init_gain: scaling factor for normal, xavier and orthogonal.
        
        :param no_dropout: no dropout for the generator
        :param direction: AtoB or BtoA
        
        :param beta1: momentum term of adam
        :param lr: initial learning rate for adam
        :param gan_mode: the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        :param pool_size: the size of image buffer that stores previously generated images 
        
        :param gpu_ids: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
        :param checkpoints_dir: models are saved here
        :param name: name of the experiment. It decides where to store samples and models
        :param preprocess: scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        :param lr_policy: learning rate policy. [linear | step | plateau | cosine]
        :param lr_decay_iters: multiply by a gamma every lr_decay_iters iterations
        """""
        self.isTrain = True
        BaseModel.__init__(self, gpu_ids=gpu_ids, isTrain=self.isTrain,
                           checkpoints_dir=checkpoints_dir, name=name,
                           preprocess=preprocess, lr_policy=lr_policy, lr_decay_iters=lr_decay_iters)
        self.direction = direction
        self.lambda_B = lambda_B
        self.lambda_A = lambda_A
        self.lambda_identity = lambda_identity

        self.input_nc = input_nc
        self.output_nc = output_nc

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(input_nc, output_nc, ngf, netG, norm,
                                        not no_dropout, init_type, init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(output_nc, input_nc, ngf, netG, norm,
                                        not no_dropout, init_type, init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(output_nc, ndf, netD,
                                            n_layers_D, norm, init_type, init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(input_nc, ndf, netD,
                                            n_layers_D, norm, init_type, init_gain, self.gpu_ids)

        if self.isTrain:
            if lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (input_nc == output_nc)
            self.fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
