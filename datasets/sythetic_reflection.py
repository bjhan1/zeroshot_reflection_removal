"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torch.utils.data as data
import glob
import scipy.stats as st
import math
import torch.nn as nn

def get_processing_kernel(kernel_size=3, sigma=2, shift_kernel_size=5):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, 1, kernel_size, kernel_size)
    # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddingl = (kernel_size - 1) // 2
    paddingr = kernel_size - 1 - paddingl
    pad = torch.nn.ReplicationPad3d((paddingl, paddingr, paddingl, paddingr, 0, 0))
    gaussian_filter = nn.Conv3d(in_channels=1, out_channels=1,
                                kernel_size=(1,kernel_size,kernel_size), bias=False)

    # print(gaussian_filter.weight.data.shape)
    gaussian_filter.weight.data = gaussian_kernel.cuda()
    gaussian_filter.weight.requires_grad = False


    shifting_filter = nn.Conv3d(in_channels=1, out_channels=1,
                                kernel_size=(1,shift_kernel_size,shift_kernel_size), bias=False)
    shifting_kernel = torch.zeros(1, 1, 1, shift_kernel_size, shift_kernel_size)
    ind1 = int(random.random() * (shift_kernel_size ** 2))
    ind2 = int(random.random() * (shift_kernel_size ** 2))
    shifting_kernel = shifting_kernel.view(-1)
    intensity = random.random()  * 0.3 + 0.2
    shifting_kernel[ind1] = 0.5
    shifting_kernel[ind2] = 0.5
    shifting_kernel = shifting_kernel.view(1, 1, 1, shift_kernel_size, shift_kernel_size)
    shifting_filter.weight.data = shifting_kernel.cuda()
    shifting_filter.weight.requires_grad = False

    s_paddingl = (shift_kernel_size - 1) // 2
    s_paddingr = shift_kernel_size - 1 - s_paddingl
    s_pad = torch.nn.ReplicationPad3d((s_paddingl, s_paddingr, s_paddingl, s_paddingr, 0, 0))


    return nn.Sequential(pad, gaussian_filter), nn.Sequential(s_pad, shifting_filter)



def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel

class SynDataIBCLN:
    def __init__(self):
        g_mask = gkern(560, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        self.g_mask = torch.tensor(g_mask.transpose(2, 0, 1))
        self.sigma = np.linspace(1, 5, 20)  
        self.max_shift =  30
        
    def __call__(self, t: torch.Tensor, r: torch.Tensor, r_label: torch.Tensor):
        
        t = t.pow(2.2)
        r = r.pow(2.2)
        
        r_max = torch.max(r.view(1,-1), dim=1, keepdim=True)[0]
        r = r + r_max.view(1,1,1,1)
        
        device = t.device
        gray = torch.mean(r, dim=1).view(-1)
        index = r_label.view(-1).long()
        sp_num = r_label.max() + 1

        sp_int = torch.zeros(sp_num).to(device).index_add(dim=0, index=index, source=gray)
        sp_size = torch.zeros(sp_num).to(device).index_add(dim=0, index=index, source=torch.ones_like(gray))
        sp_int = sp_int / sp_size.clamp(min=0.01)
        
        sp_selected = torch.argsort(sp_int, descending=True, dim=0)[:int(0.3*len(sp_int))]
        if gray[sp_selected[-1]] < 0.5:
            sp_selected = torch.nonzero(sp_int>0.5)[:,0]
        
        pix_mask = torch.zeros((t.shape[2]*t.shape[3],)).to(t.device)
        for i in range(len(sp_selected)):
            pix_mask[index == sp_selected[i]] = 1
        pix_mask = pix_mask.view(1,1,t.shape[2], t.shape[3])

        r = r * pix_mask 

        sigma = self.sigma[np.random.randint(0, len(self.sigma))]
        att = 1.0 + np.random.random() * 0.1
        # att = np.random.random() * 0.2 + 0.8
        # alpha2 = 1 - np.random.random() / 5.0
        g_sz = int(2 * np.ceil(2 * sigma) + 1)
        s_sz = (np.random.randint(0, self.max_shift-15) // 2 * 2) + 15
        
        g_kernel, s_kernel = get_processing_kernel(g_sz, sigma, s_sz)
        r_blur = g_kernel(r.unsqueeze(1)).squeeze(1).float()
        r_blur = s_kernel(r_blur.unsqueeze(1)).squeeze(1).float()
        
        pix_mask = g_kernel(pix_mask.unsqueeze(1)).squeeze(1).float()
        pix_mask = s_kernel(pix_mask.unsqueeze(1)).squeeze(1).float()
        # pix_mask = (pix_mask>0.0001).float()
        blend = r_blur + t

        maski = (blend > 1).float()
        mean_i = torch.clamp(torch.sum(blend * maski, dim=(1, 2, 3), keepdims=True) / (torch.sum(maski, dim=(1, 2, 3), keepdims=True) + 1e-6),
                             min=1)
        r_blur = r_blur - (mean_i - 1) * att
        r_blur = r_blur.clamp(min=0, max=1)

        # h, w = r_blur.shape[2:4]
        # neww = np.random.randint(0, 560 - w - 10)
        # newh = np.random.randint(0, 560 - h - 10)
        # alpha1 = self.g_mask[:, newh:newh + h, neww:neww + w].unsqueeze_(0)

        # r_blur_mask = r_blur * alpha1
        # r_blur = r_blur 
        blend = r_blur + t

        t = t.pow(1 / 2.2)
        r_blur = r_blur.pow(1 / 2.2)
        blend = blend.clamp(min=0, max=1).pow(1 / 2.2)
        

        # import torchvision
        # torchvision.utils.save_image(blend, 'blend.png')
        # torchvision.utils.save_image(r_blur, 'refle.png')
        # torchvision.utils.save_image(t, 'trans.png')
        # assert False
        return  blend.float(), r_blur.float(), pix_mask

class SyntheticDataset(data.Dataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, file_path):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        super(SyntheticDataset, self).__init__()

        self.dir_trans  = os.path.join('../../Proposed-IBCLN/datasets/reflection', 'trainA1')
        self.dir_refle  = os.path.join('./data', 'refle')
        self.dir_refle_label  = os.path.join('./data', 'refle_label')
        
        self.trans_paths  = glob.glob(os.path.join(self.dir_trans, '*.png')) \
                          + glob.glob(os.path.join(self.dir_trans, '*.jpg'))
        self.refle_paths  = glob.glob(os.path.join(self.dir_refle, '*.png')) \
                          + glob.glob(os.path.join(self.dir_refle, '*.png'))
        self.refle_label_paths  = glob.glob(os.path.join(self.dir_refle_label, '*.png')) \
                          + glob.glob(os.path.join(self.dir_refle_label, '*.png'))


        self.trans_size = len(self.trans_paths)  # get the size of dataset A1
        self.refle_size = len(self.refle_paths)  # get the size of dataset A2

        ## Define image transformer

        # self.transform1 = transforms.Compose([
        #                 #    transforms.RandomCrop(256),
        #                 #    transforms.RandomHorizontalFlip(),
        #                    transforms.ToTensor(),
        #                 ])
        
        # self.transform1 = transforms.Compose([
        #                    transforms.Resize(256),
        #                 #    transforms.RandomHorizontalFlip(),
        #                    transforms.ToTensor(),
        #                 ])
        
        self.transform = transforms.Compose([
                        #    transforms.Resize(256, InterpolationMode='nearest'),
                        #    transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                        ])
        # self.synthesize = synthesize
        # if synthesize:
        #     self.synthesizer = SynDataIBCLN()
    


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        ## Trans image
        
        trans_index = int(random.random()*self.trans_size)
        trans_path  = self.trans_paths[trans_index] 
        trans_img   = Image.open(trans_path).convert('RGB')
        trans_img  = self.transform(trans_img)
        
        ## Reflection image
        
        refle_path  = self.refle_paths[index]
        refle_img   = Image.open(refle_path).convert('RGB')
        refle_img  = self.transform(refle_img)
        
        
        refle_label_path  = self.refle_label_paths[index]
        refle_label_img   = Image.open(refle_label_path).convert('RGB')
        
        refle_label_img  = self.transform(refle_label_img)
        refle_label_img = refle_label_img[[0]]
        refle_label_img = (refle_label_img * 255).long()

        # print(refle_label_img)
        # assert False
        return trans_img, refle_img, refle_label_img
        
                
    def __len__(self):
        return 5000



class SyntheticDataset2(data.Dataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, file_path, image_crop_size=256, max_size=5000):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        super(SyntheticDataset2, self).__init__()

        self.dir_glass  = os.path.join(file_path, 'glass')
        self.dir_trans  = os.path.join(file_path, 'trans')
        self.dir_refle  = os.path.join(file_path, 'refle')
        self.dir_refle_mask  = os.path.join(file_path, 'refle_mask')

        self.glass_paths  = glob.glob(os.path.join(self.dir_glass, '*.png')) \
                          + glob.glob(os.path.join(self.dir_glass, '*.jpg'))
        self.trans_paths  = glob.glob(os.path.join(self.dir_trans, '*.png')) \
                          + glob.glob(os.path.join(self.dir_trans, '*.jpg'))
        self.refle_paths  = glob.glob(os.path.join(self.dir_refle, '*.png')) \
                          + glob.glob(os.path.join(self.dir_refle, '*.png'))
        self.refle_mask_paths  = glob.glob(os.path.join(self.dir_refle_mask, '*.png')) \
                          + glob.glob(os.path.join(self.dir_refle_mask, '*.png'))
                          
        self.glass_paths = self.glass_paths[:max_size]
        self.trans_paths = self.trans_paths[:max_size]
        self.refle_paths = self.refle_paths[:max_size]

        self.glass_size = len(self.glass_paths) 
        self.trans_size = len(self.trans_paths) 
        self.refle_size = len(self.refle_paths) 

        ## Define image transformer
        self.transform = transforms.Compose([
                           transforms.ToTensor(),
                        ])
        self.crop_size = image_crop_size
    def crop(self, tensor, h1,w1,h2,w2):
        nc, h, w = tensor.shape
        assert h1>=0 and h1<=h, "h1 is out of image size"
        assert w1>=0 and w1<=w, "w1 is out of image size"
        assert h2>=0 and h2<=h, "h2 is out of image size"
        assert w2>=0 and w2<=w, "w2 is out of image size"
        return tensor[:, h1:h2, w1:w2]
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        index = index % self.glass_size
        
        ## Glass image
        glass_path  = self.glass_paths[index]
        glass_img   = Image.open(glass_path).convert('RGB')
        glass_img  = self.transform(glass_img)

        ## Trans image
        trans_path  = self.trans_paths[index] 
        trans_img   = Image.open(trans_path).convert('RGB')
        trans_img  = self.transform(trans_img)
        
        ## Reflection image
        refle_path  = self.refle_paths[index]
        refle_img   = Image.open(refle_path).convert('RGB')
        refle_img  = self.transform(refle_img)

        # ## Reflection image
        # refle_mask_path  = self.refle_mask_paths[index]
        # refle_mask_img   = Image.open(refle_mask_path).convert('L')
        # refle_mask_img  = self.transform(refle_mask_img)
        
        crop_size=self.crop_size

        ## Resize if the input image is small
        if glass_img.shape[1] < glass_img.shape[2]:
            if glass_img.shape[1] < crop_size :
                w2 = int(float(crop_size) / float(glass_img.shape[1]))
                h2 = crop_size
                glass_img = F.interpolate(glass_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                trans_img = F.interpolate(trans_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                refle_img = F.interpolate(refle_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                # refle_mask_img = F.interpolate(refle_mask_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
        else:
            if glass_img.shape[2] < crop_size :
                h2 = int(float(crop_size) / float(glass_img.shape[2]))
                w2 = crop_size
                glass_img = F.interpolate(glass_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                trans_img = F.interpolate(trans_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                refle_img = F.interpolate(refle_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
                # refle_mask_img = F.interpolate(refle_mask_img.unsqueeze(0), (h2, w2), mode='bilinear', align_corners=False).squeeze(0)
        
        ## Crop
        nc, h, w = glass_img.shape
        h1 = random.randint(0, h - crop_size)
        w1 = random.randint(0, w - crop_size)
        h2 = h1 + crop_size
        w2 = w1 + crop_size
        glass_img = self.crop(glass_img, h1,w1,h2,w2)
        trans_img = self.crop(trans_img, h1,w1,h2,w2)
        refle_img = self.crop(refle_img, h1,w1,h2,w2)
        # refle_mask_img = self.crop(refle_mask_img, h1,w1,h2,w2)

        assert glass_img.shape[1]==crop_size
        assert glass_img.shape[2]==crop_size

        if random.random()>0.5:
            glass_img = torch.flip(glass_img, dims=[2])
            trans_img = torch.flip(trans_img, dims=[2])
            refle_img = torch.flip(refle_img, dims=[2])
            # refle_mask_img = torch.flip(refle_mask_img, dims=[2])

        return glass_img, trans_img, refle_img
        
    def __len__(self):
        return self.glass_size



class ReflectionTesetset(data.Dataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, file_path):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        super(ReflectionTesetset, self).__init__()

        if 'WildSceneDataset_sub' in file_path:
            child_t = 'transmission'
            child_g = 'glass'
            db_name = 'SIR'
            child_m = 'mask'
        else:
            raise ValueError('%s is not available test set'%db_name)

        self.dir_glass  = os.path.join(file_path, child_g)
        self.dir_trans  = os.path.join(file_path, child_t)
        self.dir_mask  = os.path.join(file_path, child_m)
        
        self.glass_paths  = glob.glob(os.path.join(self.dir_glass, '*.png')) \
                          + glob.glob(os.path.join(self.dir_glass, '*.jpg'))
        self.trans_paths  = glob.glob(os.path.join(self.dir_trans, '*.png')) \
                          + glob.glob(os.path.join(self.dir_trans, '*.jpg'))
        self.mask_paths  = glob.glob(os.path.join(self.dir_mask, '*.png')) \
                          + glob.glob(os.path.join(self.dir_mask, '*.jpg'))
        
        self.glass_size = len(self.glass_paths) 
        self.trans_size = len(self.trans_paths) 
        
        ## Define image transformer
        self.transform = transforms.Compose([
                           transforms.ToTensor(),
                        ])
                        
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        index = index % self.glass_size
        
        ## Glass image
        glass_path  = self.glass_paths[index]
        glass_img   = Image.open(glass_path).convert('RGB')
        glass_img  = self.transform(glass_img)

        ## Trans image
        trans_path  = self.trans_paths[index] 
        trans_img   = Image.open(trans_path).convert('RGB')
        trans_img  = self.transform(trans_img)
        

        ## Mask image
        mask_path  = self.mask_paths[index] 
        mask_img   = Image.open(trans_path.replace('transmission', 'mask')).convert('RGB')
        mask_img  = self.transform(mask_img)

        height, width = trans_img.shape[-2:]
        height = (height // 16) * 19
        width = (width // 16) * 19
        glass_img = torch.nn.functional.interpolate(glass_img.unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0)
        trans_img = torch.nn.functional.interpolate(trans_img.unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0)
        mask_img = torch.nn.functional.interpolate(mask_img.unsqueeze(0), (height, width), mode='nearest').squeeze(0)
        filename = self.glass_paths[index].split('/')[-1]
        
        return glass_img, trans_img, mask_img, filename
        
    def __len__(self):
        return self.glass_size




class GeneralDataset(data.Dataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, data_dir):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        super(GeneralDataset, self).__init__()
        self.data_dir  = data_dir
        self.data_paths  = glob.glob(os.path.join(self.data_dir, '*.png')) \
                          + glob.glob(os.path.join(self.data_dir, '*.jpg'))
        self.dataset_size = len(self.data_paths) 
        
        ## Define image transformer
        self.transform = transforms.Compose([transforms.Resize(512),
                                            transforms.ToTensor()])
          
    def __getitem__(self, index):
        ## image
        data_paths  = self.data_paths[index]
        data_img   = Image.open(data_paths).convert('RGB')
        data_img  = self.transform(data_img)
        # data_img = data_img/255.0
        return data_img
        
    def __len__(self):
        return self.dataset_size

