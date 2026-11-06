import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

# Perceptual loss on VGG16 features (--lambda_VGG), used alongside the adversarial and NCE terms.
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()  # relu3_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg
        self.device = device

        # ImageNet mean and std. Inputs in [-1, 1] are normalized with them without a prior rescale.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))


    def forward(self, x, y):
        # L1 distance between VGG features of generated patches (x) and of the unpaired aggrecan
        # patches of the same batch (y).
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.l1_loss(x_vgg, y_vgg)

class VirtualPNNModel(BaseModel):
    """ VirtualPNN: the CUT model (Park, Efros, Zhang and Zhu, ECCV 2020) with two added generator
    terms, a VGG16 perceptual loss and an anti-vessel loss on vessel-only H&E patches. The
    adversarial, NCE and perceptual terms use only the PNN patches of a batch.
    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lr_G', type=float, default=None, help='learning rate for generator (overrides --lr if set)')
        parser.add_argument('--lr_D', type=float, default=None, help='learning rate for discriminator (overrides --lr if set)')
        parser.add_argument('--lambda_VGG', type=float, default=1.0, help='weight for VGG perceptual loss')
        parser.add_argument('--use_vessel_patches', action='store_true',
                            help='Include vessel-only patches (H&E with no PNNs) during training for negative supervision')
        parser.add_argument('--vessel_patch_ratio', type=float, default=0.25,
                            help='Probability of sampling a vessel-only patch instead of a regular PNN patch (0 to 1)')
        parser.add_argument('--lambda_anti_vessel', type=float, default=10.0,
                    help='Loss weight for suppressing staining on vessel-only patches (target = white)')


        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        # Self-regularization is disabled: calculate_SR_loss is a signed mean and is not a usable regularizer.
        # A non-zero --self_regularization raises.
        parser.add_argument('--self_regularization', type=float, default=0.0,
                            help='disabled; a non-zero value raises')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']

        self.loss_names.append("VGG")  # Add VGG to the loss printouts
        self.loss_names.append("anti_vessel")


        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
            if opt.lambda_NCE > 0:
                self.model_names.append('F')
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)


        if opt.lambda_NCE > 0:
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, 
                                        opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))


     

     
            self.vgg_loss_fn = VGGPerceptualLoss(self.device)
            self.lambda_vgg = opt.lambda_VGG # exposed
              

            lr_G = opt.lr_G if opt.lr_G is not None else opt.lr
            lr_D = opt.lr_D if opt.lr_D is not None else opt.lr

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate gradients for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        d_loss = self.compute_D_loss()
        if d_loss is not None:
            d_loss.backward()
            self.optimizer_D.step()

        # update G (+ F if used)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.lambda_NCE > 0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.optimizer_G.step()
        if self.opt.lambda_NCE > 0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if input.get('type', 'pnn') == 'vessel':
            self.real_B = torch.ones_like(self.real_A).to(self.device)
        else:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_type = input.get('type', 'pnn')  # vessel or pnn
        # Normalize to a list of per-sample tags
        if isinstance(self.image_type, str):
            self.image_type = [self.image_type] * self.real_A.size(0)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        pnn_indices = [i for i,t in enumerate(self.image_type) if t != 'vessel']
        if len(pnn_indices) == 0:
            # ensure log-friendly zeros
            z = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = z
            self.loss_D_real = z
            self.loss_D      = z
            return None  # tell caller to skip D backward/step

        fake_pnn  = self.fake_B.detach()[pnn_indices]
        realB_pnn = self.real_B[pnn_indices]

        pred_fake = self.netD(fake_pnn)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD(realB_pnn)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        return self.loss_D



    def compute_G_loss(self):

        pnn_indices = [i for i, t in enumerate(self.image_type) if t != 'vessel']
        vessel_indices = [i for i, t in enumerate(self.image_type) if t == 'vessel']

        """Calculate GAN and NCE loss for the generator"""
        # GAN loss on PNN patches only
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0 and len(pnn_indices) > 0:
            pred_fake_pnn = self.netD(self.fake_B[pnn_indices])
            self.loss_G_GAN = self.criterionGAN(pred_fake_pnn, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = torch.tensor(0.0, device=self.device)

        # NCE loss on PNN patches only
        
        if self.opt.lambda_NCE > 0.0 and len(pnn_indices) > 0:
            try:
                real_A_pnn = self.real_A[pnn_indices]
                fake_B_pnn = self.fake_B[pnn_indices]
                self.loss_NCE = self.calculate_NCE_loss(real_A_pnn, fake_B_pnn)
            except Exception as e:
                print(f"[WARNING] NCE loss failed: {e}")
                self.loss_NCE = torch.tensor(0.0, device=self.device)
        else:
            self.loss_NCE = torch.tensor(0.0, device=self.device)

        if self.opt.self_regularization > 0.0:
            raise RuntimeError(
                "--self_regularization is disabled: calculate_SR_loss returns a signed mean, "
                "so minimising it is unbounded and does not pull the output towards the input. "
                "No released model used it. Rewrite the loss before enabling it.")
        self.loss_SR = torch.tensor(0.0, device=self.device)

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0 and len(pnn_indices) > 0:
            try:
                real_B_pnn = self.real_B[pnn_indices]
                idt_B_pnn = self.idt_B[pnn_indices]
                self.loss_NCE_Y = self.calculate_NCE_loss(real_B_pnn, idt_B_pnn)
            except Exception as e:
                print(f"[WARNING] NCE_Y loss failed: {e}")
                self.loss_NCE_Y = torch.tensor(0.0, device=self.device)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
            self.loss_NCE_Y = torch.tensor(0.0, device=self.device)

        

        if self.opt.lambda_VGG > 0.0 and len(pnn_indices) > 0:
            try:
                fake_B_pnn = self.fake_B[pnn_indices]
                real_B_pnn = self.real_B[pnn_indices]
                self.loss_VGG = self.lambda_vgg * self.vgg_loss_fn(fake_B_pnn, real_B_pnn)
            except Exception as e:
                print(f"[WARNING] VGG loss failed: {e}")
                self.loss_VGG = torch.tensor(0.0, device=self.device)
        else:
            self.loss_VGG = torch.tensor(0.0, device=self.device)

        
        if self.opt.lambda_anti_vessel > 0.0:
            self.loss_anti_vessel = self.calculate_anti_vessel_loss(self.fake_B, self.image_type)
        else:
            self.loss_anti_vessel = torch.tensor(0.0, device=self.device)

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SR + self.loss_VGG + self.loss_anti_vessel
        print("Loss values:")
        print(f"G_GAN: {self.loss_G_GAN.item()}, NCE: {self.loss_NCE.item()}, VGG: {self.loss_VGG.item()}, Anti-vessel: {self.loss_anti_vessel.item()}")

        return self.loss_G


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_SR_loss(self,src,tgt):
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2
    
    def calculate_anti_vessel_loss(self, fake_B, image_type):
        vessel_indices = [i for i, t in enumerate(image_type) if t == 'vessel']
        if not vessel_indices:
            return torch.tensor(0.0, device=self.device)

        fake_vessel = fake_B[vessel_indices]

        if torch.isnan(fake_vessel).any() or torch.isinf(fake_vessel).any():
            print(f"[ERROR] Anti-vessel loss is NaN! fake_vessel stats: min={fake_vessel.min().item()}, max={fake_vessel.max().item()}")
            return torch.tensor(0.0, device=self.device)

        diff = torch.mean(torch.abs(fake_vessel - 1.0))

        loss = self.opt.lambda_anti_vessel * diff

        if torch.isnan(loss).any():
            print(f"[ERROR] Anti-vessel loss is NaN! fake_vessel stats: min={fake_vessel.min().item()}, max={fake_vessel.max().item()}")
            return torch.tensor(0.0, device=self.device)

        return loss


