import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torch



class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        if getattr(opt, "use_vessel_patches", False):
            self.vessel_A_paths = sorted(make_dataset(os.path.join(opt.dataroot, 'trainA_vessel'), opt.max_dataset_size))

        ###


        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B


    def __getitem__(self, index):
        use_vessel = getattr(self.opt, "use_vessel_patches", False) and (random.random() < self.opt.vessel_patch_ratio)

        if use_vessel:
            # Load vessel-only patch
            A_path = self.vessel_A_paths[index % len(self.vessel_A_paths)]
            A_img = Image.open(A_path).convert('RGB')
            transform = get_transform(self.opt)
            A = transform(A_img)
            B = torch.full_like(A, fill_value=1.0, dtype=torch.float32)
            return {
                'A': A,
                'B': B,  # No PNN stain
                'type': 'vessel',
                'A_paths': A_path,
                'B_paths': A_path
            }

        else:
            # Normal paired PNN patch
            A_path = self.A_paths[index % self.A_size]
            index_B = index % self.B_size if self.opt.serial_batches else random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            transform = get_transform(self.opt)
            A = transform(A_img)
            B = transform(B_img)

            return {
                'A': A,
                'B': B,
                'type': 'pnn',
                'A_paths': A_path,
                'B_paths': B_path
            }
            ###
    def __len__(self):
        base_len = max(self.A_size, self.B_size)
        if getattr(self.opt, "use_vessel_patches", False):
            return max(base_len, len(self.vessel_A_paths))
        return base_len
    ###
