import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import pytorch_ssim
import numpy as np
import math 
import torch.optim as optim
import cv2
import functools
import random 


def neighboring(x, kernel_size=5):
    assert kernel_size % 2 == 1
    p = kernel_size//2
    neighboring_kernel = torch.ones(1,1,1,kernel_size,kernel_size).float().to(x.device)
    neighboring_kernel /= neighboring_kernel.sum()
    return F.conv3d(F.pad(x, (p,p,p,p), 'replicate').unsqueeze(1), neighboring_kernel).squeeze(1)



def enhanced_gradient(img):
    color_grad_layer = GradientConcatLayer().to(img.device)
    img_grad_rev = color_grad_layer(img, single_scale=True)
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        img_grad2 = color_grad_layer(img**gamma, single_scale=True)
        img_grad2_mag = torch.sum(torch.abs(img_grad2), dim=1, keepdim=True).repeat(1, img_grad2.shape[1], 1, 1)
        img_grad_rev_mag = torch.sum(torch.abs(img_grad_rev), dim=1, keepdim=True).repeat(1, img_grad_rev.shape[1], 1, 1)
        img_grad_rev[img_grad_rev_mag < img_grad2_mag] = img_grad2[img_grad_rev_mag < img_grad2_mag]
    return img_grad_rev.detach().clone()


def key_checking(key, key_list):
    for key_cmp in key_list:
        if key_cmp in key:
            # print(key)
            return True
    return False


def load_test_data(data_dir, output_size=(512,512)):
    
    Glass = cv2.imread('%s/glass.png' % data_dir)
    Glass = cv2.cvtColor(Glass, cv2.COLOR_BGR2RGB)
    Glass = Glass.astype(float)/255
    Glass = torch.tensor(Glass).permute(2,0,1).unsqueeze(0)
    Glass = Glass.float()
    Glass = F.interpolate(Glass, output_size, mode='bilinear', align_corners=False)

    Refer = cv2.imread('%s/reference0044.png' % data_dir)
    Refer = cv2.cvtColor(Refer, cv2.COLOR_BGR2RGB)
    Refer = Refer.astype(float)/255
    Refer = torch.tensor(Refer).permute(2,0,1).unsqueeze(0)
    Refer = Refer.float()
    Refer = F.interpolate(Refer, output_size, mode='bilinear', align_corners=False)

    X, Y= torch.meshgrid(torch.linspace(-1, 1,output_size[0]), torch.linspace(-1, 1,output_size[0]))
    grid = torch.cat((Y.unsqueeze(2), X.unsqueeze(2)), 2)
    grid = grid.unsqueeze(0)
    ReferGrids = []
    ReferGridsInv = []
    for k in range(60):
        scale = 1.0-0.01 * float(k)
        if scale < 0.1:
            break
        ReferGrids.append(grid * scale)
        ReferGridsInv.append(grid / scale)
    ReferGridsSeq = torch.cat(ReferGrids, 0)
    ReferGridsInvSeq = torch.cat(ReferGridsInv, 0)
    ReferSeq = F.grid_sample(Refer.repeat(len(GridsSeq),1,1,1), GridsSeq, align_corners=False)
    
    return Glass, ReferSeq, GridsSeq, GridsInvSeq


def load_test_data2(data_dir, output_size=(512,512)):
    
    Glass = cv2.imread('%s/glass.png' % data_dir)
    Glass = cv2.cvtColor(Glass, cv2.COLOR_BGR2RGB)
    Glass = Glass.astype(float)/255
    Glass = torch.tensor(Glass).permute(2,0,1).unsqueeze(0)
    Glass = Glass.float()
    Glass = F.interpolate(Glass, output_size, mode='bilinear', align_corners=False)

    Refer = cv2.imread('%s/reference.png' % data_dir)
    Refer = cv2.cvtColor(Refer, cv2.COLOR_BGR2RGB)
    Refer = Refer.astype(float)/255
    Refer = torch.tensor(Refer).permute(2,0,1).unsqueeze(0)
    Refer = Refer.float()
    Refer = F.interpolate(Refer, output_size, mode='bilinear', align_corners=False)

    X, Y= torch.meshgrid(torch.linspace(-1, 1,output_size[0]), torch.linspace(-1, 1,output_size[1]))
    grid = torch.cat((Y.unsqueeze(2), X.unsqueeze(2)), 2)
    grid = grid.unsqueeze(0)
    Grids = []
    GridsInv = []
    for k in range(60):
        scale = 1.0-0.01 * float(k)
        if scale < 0.1:
            break
        Grids.append(grid * scale)
        GridsInv.append(grid / scale)
    GridsSeq = torch.cat(Grids, 0)
    GridsInvSeq = torch.cat(GridsInv, 0)
    ReferSeq = F.grid_sample(Refer.repeat(len(GridsSeq),1,1,1), GridsSeq, align_corners=False)
    GlassSeq = F.grid_sample(Glass.repeat(len(GridsSeq),1,1,1), GridsSeq, align_corners=False)
    
    return GlassSeq, ReferSeq, GridsSeq, GridsInvSeq

def cvtXYZ2Sph(x,y,z):
    r = torch.sqrt(x*x + y*y + z*z).clamp(1e-12)
    x = x / r
    y = y / r
    z = z / r
    # theta = math.pi - torch.arccos(z)
    theta = torch.arccos(z) # Vertical
    # phi = torch.arctan(torch.divide(y, x, out=torch.ones_like(y)*1000, where=x!=0))
    x[torch.abs(x) < 1e-9] = 1e-9
    phi = torch.arctan(y/x) # Horizontal 
    phi[x<0] = math.pi + phi[x<0]
    phi[torch.logical_and(x>0, y<0)] = math.pi *2 + phi[torch.logical_and(x>0, y<0)]
    
    return [phi, theta, r]

def cvtSph2Pix(phi, theta, H, W):
    w_array = phi /(math.pi*2)* W  # horizontal
    h_array = theta /(math.pi) *H # Vertical
    return [w_array, h_array]

def cvtXYZ2Pix(x,y,z, H, W):
    [phi, theta, r]= cvtXYZ2Sph(x, y, z)
    [w_array, h_array] = cvtSph2Pix(phi, theta, H, W)
    w_array = w_array.long()
    h_array = h_array.long()
    return [h_array, w_array]

def getGlassRegionXYZ(window_size, output_size):
    y_arr = torch.linspace(window_size, -window_size, output_size)
    z_arr = torch.linspace(window_size, -window_size, output_size)
    Z, Y = torch.meshgrid(z_arr, y_arr)
    # Y: vertical Z: horizontal
    X = -torch.ones_like(Y)
    XYZ = torch.stack([X,Y,Z], dim=0)
    XYZ /= XYZ.norm(dim=0, keepdim=True).clamp(1e-12)
    return XYZ

def gatherColorbyXYZ(XYZ, pano):
    X, Y, Z = XYZ[0],  XYZ[1],  XYZ[2]
    h_array, w_array = cvtXYZ2Pix(X, Y, Z, pano.shape[2], pano.shape[3])
    tmp = pano.squeeze(0)
    R, G, B = tmp[0], tmp[1], tmp[2]
    h_array= h_array.clamp(0, pano.shape[2]-1)
    w_array= w_array.clamp(0, pano.shape[3]-1)
    r = R[h_array, w_array]
    g = G[h_array, w_array]
    b = B[h_array, w_array]
    
    patch = torch.stack([r,g,b], 0).unsqueeze(0)
    return patch
    

def getHouseholderMatrix(h):
    # ax + by + cz + d = 0 
    a, b, c,d  = h[0], h[1], h[2], h[3]
    n = math.sqrt(a * a + b * b + c * c)
    a /=n
    b /=n
    c /=n
    A = torch.tensor([[1-2*a*a , -2*a*b, -2*a*c, -2*a*d],[-2*a*b , 1-2*b*b, -2*b*c, -2*b*d],[-2*a*c , -2*b*c, 1-2*c*c, -2*c*d], [0,0,0,1]])
    return A

def reflecting_xyz(XYZ, dist, Hm, return_scale=False):
    assert len(XYZ.shape) == 3
    nc, height, width = XYZ.shape
    XYZ = XYZ.view(nc, -1)
    if len(XYZ) == 3:
        XYZ = torch.cat([XYZ, torch.zeros(1,height*width)], dim=0)
    origin = torch.tensor([[0.0],[0.0],[0.0],[1.0]]).float().to(XYZ.device)
    
    XYZ_transformed = torch.mm(Hm, dist*XYZ-(dist-1)*origin)
    XYZ_transformed = XYZ_transformed.reshape(4, height, width)
    XYZ_transformed = XYZ_transformed[:3]
    if return_scale:
        XYZ_scaled = dist*XYZ.reshape(4,height, width)
        return XYZ_transformed, XYZ_scaled[:3]
    else:
        return XYZ_transformed




def reflecting(XYZ, img, dist, Hm):
    assert len(XYZ.shape) == 3
    nc, height, width = XYZ.shape
    XYZ = XYZ.view(nc, -1)
    if len(XYZ) == 3:
        XYZ = torch.cat([XYZ, torch.zeros(1,height*width)], dim=0)
    
    origin = torch.tensor([[0.0],[0.0],[0.0],[1.0]]).float().to(XYZ.device)
    
    XYZ_transformed = torch.mm(Hm, dist*XYZ-(dist-1)*origin)
    XYZ_transformed = XYZ_transformed.reshape(4, height, width)
    XYZ_transformed = XYZ_transformed[:3]
    refer_patch_img = gatherColorbyXYZ(XYZ_transformed, img)
    return refer_patch_img

def projecting_to_plane(XYZ, h):
    XYZ = XYZ[:3]
    h = torch.tensor(h).to(XYZ.device)
    h = h.view(-1,1,1)
    value = torch.sum(XYZ * h, dim=0, keepdim=True)
    
    value[torch.abs(value) < 1e-12] = 1e-12
    distance = 1.0/value
    XYZ = XYZ * torch.abs(distance)
    
    return XYZ


def projecting_to_plane_rev(XYZ, h):
    XYZ = XYZ[:,:3,:,:]
    h = torch.tensor(h).to(XYZ.device)
    h = h.view(1,-1,1,1)
    value = torch.sum(XYZ * h, dim=1, keepdim=True)
    
    value[torch.abs(value) < 1e-12] = 1e-12
    distance = 1.0/value
    XYZ = XYZ * torch.abs(distance)
    
    return XYZ


def projecting_to_adjacent_plane(XYZ):
    XYZ = XYZ[:,:3,:,:]
    XYZ /= XYZ.norm(dim=1,keepdim=True)
    h = torch.mean(XYZ, dim=0, keepdim=True)
    h /= h.norm(dim=1, keepdim=True)
    unit_sphere_radius = 1.0
    value = torch.sum(XYZ * h, dim=1, keepdim=True)
    
    value[torch.abs(value) < 1e-12] = 1e-12
    distance = unit_sphere_radius/value
    XYZ = XYZ * torch.abs(distance)
    
    return XYZ

def gatherGlassColorbyXYZ(reflected_refer_xyz, Glass, glass_plane=[1.0, 0.0, 0.0], glass_size=1, glass_image_size=512):
    reflected_refer_xyz_plane = projecting_to_plane(reflected_refer_xyz, glass_plane)
    reflected_refer_y = reflected_refer_xyz_plane[1]
    reflected_refer_z = reflected_refer_xyz_plane[2]
    reflected_refer_w = (glass_size-reflected_refer_y) / 2 * glass_image_size
    reflected_refer_h = (glass_size-reflected_refer_z) / 2 * glass_image_size
    reflected_refer_h = reflected_refer_h.long().clamp(0,glass_image_size-1)
    reflected_refer_w = reflected_refer_w.long().clamp(0,glass_image_size-1)
    Glass2 = Glass[:,:,reflected_refer_h,reflected_refer_w]
    return Glass2


def gatherGlassColorbyXYZ_rev(reflected_refer_xyz, Glass, glass_plane=[1.0, 0.0, 0.0], glass_size=1, glass_image_size=512):
    reflected_refer_xyz_plane = projecting_to_plane_rev(reflected_refer_xyz, glass_plane)
    _, reflected_refer_y, reflected_refer_z = torch.split(reflected_refer_xyz_plane, 1,1)
    referected_refer_w = -reflected_refer_y
    referected_refer_h = -reflected_refer_z
    grids = torch.cat([referected_refer_w, referected_refer_h], dim=1).permute(0,2,3,1)
    Glass2 = F.grid_sample(Glass.expand(len(grids), -1, -1, -1), grids, align_corners=False)
    return Glass2


def reflecting_with_backward(XYZ, img, dist, dists, Hm):
    assert len(XYZ.shape) == 3
    nc, height, width = XYZ.shape
    XYZ = XYZ.view(nc, -1)
    if len(XYZ) == 3:
        XYZ = torch.cat([XYZ, torch.zeros(1,height*width)], dim=0)
    
    origin = torch.tensor([[0.0],[0.0],[0.0],[1.0]]).float().to(XYZ.device)
    XYZ_transformed = torch.mm(Hm, dist*XYZ-(dist-1)*origin)
    XYZ_transformed = XYZ_transformed.reshape(4, height, width)
    XYZ_transformed = XYZ_transformed[:3]
    XYZ_transformed_projected = XYZ_transformed / XYZ_transformed.norm(dim=0,keepdim=True).clamp(min=1e-12)

    patch_imgs = []
    dists = 1.0/np.linspace(1,0.005, 50)
    for dist2 in dists:
        patch_img = reflecting(XYZ_transformed_projected, img, dist2, Hm)
        patch_imgs.append(patch_img)

    refer_patch_img = gatherColorbyXYZ(XYZ_transformed_projected, img)
    patch_imgs = torch.cat(patch_imgs, dim=0)

    return refer_patch_img, patch_imgs


def crop_glass_reference(pano):
    height, width, depth = pano.shape
    yz_range = 2.5
    step = yz_range * 2 / 1000.
    y_arr = np.arange(-yz_range, yz_range, step)
    z_arr = np.arange(-yz_range, yz_range, step)
    Y, Z = np.meshgrid(y_arr, z_arr)
    X = -np.ones_like(Y)
    R = np.power(np.power(X, 2)+np.power(Y, 2)+np.power(Z,2), 0.5)
    X /= R
    Y /= R
    Z /= R
    phi, theta, r = cvtXYZ2Sph(X, Y, Z)
    pix_w, pix_h = cvtSph2Pix(phi, theta, height, width)
    pix_w[pix_w >= width] = pix_w[pix_w >= width] - width
    glass_patch_img = pano[pix_h.astype(int),pix_w.astype(int), :].copy()
    
    Y, Z = np.meshgrid(y_arr, z_arr)
    X = np.ones_like(Y)
    R = np.power(np.power(X, 2)+np.power(Y, 2)+np.power(Z,2), 0.5)
    X /= R
    Y /= R
    Z /= R
    phi, theta, r = cvtXYZ2Sph(X, Y, Z)
    pix_w, pix_h = cvtSph2Pix(phi, theta, height, width)
    pix_w[pix_w >= width] = pix_w[pix_w >= width] - width
    refer_patch_img = pano[pix_h.astype(int),pix_w.astype(int), :].copy()
    
    return glass_patch_img,  refer_patch_img

def criterion_restoration(source, target, alpha=1.0, clip=False):
    
    source_grad = compute_gradient(source)
    target_grad = compute_gradient(target)
    if clip:
        source = source.clamp(0,1)
        source_grad = source_grad.clamp(-1,1)

    return F.mse_loss(source, target) + alpha*F.mse_loss(source_grad, target_grad)

def compute_gradient(x):
    dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float() / 5.0
    dy_kernel = dx_kernel.clone().t()
    dx_kernel = dx_kernel.view(1, 1, 1, 3, 3).to(x.device)
    dy_kernel = dy_kernel.view(1, 1, 1, 3, 3).to(x.device)
    dx = F.conv3d(F.pad(x, (1,1,1,1), 'replicate').unsqueeze(1), dx_kernel).squeeze(1)
    dy = F.conv3d(F.pad(x, (1,1,1,1), 'replicate').unsqueeze(1), dy_kernel).squeeze(1)        
    return torch.cat((dx,dy), 1)

def criterion_exclusive_gradient(source, target, ShiftGrids):

    ## 1. Compute gradients of results
    source_gray_grad = compute_gradient(torch.mean(source,dim=1,keepdim=True))
    target_gray_grad = compute_gradient(torch.mean(target,dim=1,keepdim=True))
    
    ## 2. Gradient Matching
    target_gray_grad_shifted = F.grid_sample(target_gray_grad.repeat(len(ShiftGrids), 1, 1, 1), ShiftGrids, align_corners=False)
    target_gray_grad_max_arg = torch.max(torch.sum(torch.abs(source_gray_grad)*torch.abs(target_gray_grad_shifted),dim=1,keepdim=True), dim=0, keepdim=True)[1]
    target_gray_grad_matched = torch.gather(target_gray_grad_shifted, 0, target_gray_grad_max_arg.repeat(1, target_gray_grad.shape[1], 1, 1))
    
    ## 3. Gradient exclusive loss
    loss = torch.sum(torch.abs(source_gray_grad)*torch.abs(target_gray_grad_matched), dim=1).mean()

    return loss 


def criterion_exclusive_gradient_rev(source, target, shift):

    ## 1. Compute gradients of results
    source_gray_grad = compute_gradient(torch.mean(source,dim=1,keepdim=True))
    target_gray_grad = compute_gradient(torch.mean(target,dim=1,keepdim=True))
    
    ## 2. Gradient Matching
    target_gray_grad_shifted = [] 
    for dx in range(-shift, shift + 1):
        for dy in range(-shift, shift + 1):
            target_gray_grad_shifted.append(torch.roll(target_gray_grad, (dy, dx), dims=(2,3)))
    target_gray_grad_shifted = torch.cat(target_gray_grad_shifted,  dim=0)
    target_gray_grad_max_arg = torch.max(torch.sum(torch.abs(source_gray_grad)*torch.abs(target_gray_grad_shifted),dim=1,keepdim=True), dim=0, keepdim=True)[1]
    target_gray_grad_matched = torch.gather(target_gray_grad_shifted, 0, target_gray_grad_max_arg.repeat(1, target_gray_grad.shape[1], 1, 1))
    
    ## 3. Gradient exclusive loss
    loss = torch.sum(torch.abs(source_gray_grad)*torch.abs(target_gray_grad_matched), dim=1).mean()

    return loss 
    
# def getHouseholderMatrix(h, d):
#     # ax + by + cz + d = 0 
#     a, b, c = h[0], h[1], h[2]
#     n = math.sqrt(a * a + b * b + c * c)
#     a /=n
#     b /=n
#     c /=n
#     A = np.array([[1-2*a*a , -2*a*b, -2*a*c, -2*a*d],[-2*a*b , 1-2*b*b, -2*b*c, -2*b*d],[-2*a*c , -2*b*c, 1-2*c*c, -2*c*d], [0,0,0,1]])
#     return A

def getDistanceToPlane(h, XYZ):
    a, b, c = h[0], h[1], h[2]
    distance = XYZ[0] * a + XYZ[1] * b + XYZ[2] * c
    d = np.mean(distance)
    return d


def cvtPix2Sph(H, W, height, width):
    # phi: horizontal axis and [-180, 180]
    phi = (W/width)*360
    # theta: vertical axis and [-90, 90]
    theta = (H/height)*180
    return [torch.deg2rad(phi), torch.deg2rad(theta)]


def cvtSph2XYZ(phi, theta, r):
    # pi should be 0~2pi
    # theta shoulbe 0~pi
    x = r*torch.sin(theta)*torch.cos(phi)
    y = r*torch.sin(theta)*torch.sin(phi)
    z = r*torch.cos(theta)
    return [x,y,z]

def cvtPix2XYZ(H, W, height, width):
    [phi, theta] = cvtPix2Sph(H, W, height, width)
    r = torch.ones_like(phi)*1
    [X, Y, Z] = cvtSph2XYZ(phi, theta, r)
    return [X, Y, Z]

def getXYZcoord(image_height, image_width):
    h_array, w_array = torch.meshgrid(torch.linspace(0, image_height-1, image_height), torch.linspace(0, image_width-1, image_width))
    X, Y, Z = cvtPix2XYZ(h_array, w_array, image_height, image_width)
    XYZ = torch.stack([X,Y,Z], dim=0)
    return XYZ



def load_test_data3(data_dir, num_scales=60, output_size=(512,512)):
    # print(data_dir)
    pano = cv2.imread(data_dir)
    # print(pano)
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    pano = pano.astype(float)/255

    Glass, Refer = crop_glass_reference(pano)
    
    Glass = torch.tensor(Glass).float().permute(2,0,1).unsqueeze(0)
    Glass = F.interpolate(Glass, output_size, mode='bilinear', align_corners=False)

    Refer = torch.tensor(Refer).float().permute(2,0,1).unsqueeze(0)
    Refer = F.interpolate(Refer, output_size, mode='bilinear', align_corners=False)

    X, Y= torch.meshgrid(torch.linspace(-1, 1,output_size[0]), torch.linspace(-1, 1,output_size[1]))
    grid = torch.cat((Y.unsqueeze(2), X.unsqueeze(2)), 2)
    grid = grid.unsqueeze(0)
    Grids = []
    GridsInv = []
    scales = np.linspace(1, 0.4, num_scales)
    for scale in scales:
        Grids.append(grid * scale)
        GridsInv.append(grid / scale)
    GridsSeq = torch.cat(Grids, 0)
    GridsInvSeq = torch.cat(GridsInv, 0)
    ReferSeq = F.grid_sample(Refer.repeat(len(GridsSeq),1,1,1), GridsSeq, align_corners=False)
    GlassSeq = F.grid_sample(Glass.repeat(len(GridsSeq),1,1,1), GridsSeq, align_corners=False)
    
    return GlassSeq, ReferSeq, pano


def load_test_data4(data_dir):
    pano = cv2.imread(data_dir)
    print(data_dir)
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    pano = pano.astype(float)/255
    pano = torch.tensor(pano).permute(2,0,1).unsqueeze(0)
    return pano

def down_sample(img, factor=2):
    _, _, h, w = img.shape
    img = F.interpolate(img, (h//factor, w//factor), mode='bilinear', align_corners=False)
    return img 

class NCC(nn.Module):
    def __init__(self, patch_size):
        super(NCC, self).__init__()
        expanding_kernel = torch.zeros(3*patch_size*patch_size,1,3,patch_size, patch_size)
        cnt = 0
        for i in range(3):
            for j in range(patch_size):
                for k in range(patch_size):     
                    expanding_kernel[cnt,0,i,j,k] = 1
                    cnt += 1
        
        self.expanding_layer = nn.Conv3d(1, 3*patch_size*patch_size, (3,patch_size,patch_size), padding=(0,patch_size//2,patch_size//2), bias=False)
        self.expanding_layer.weight.data = expanding_kernel
        
    def normalize(self, tensor):
        exp_tensor = self.expanding_layer(tensor.unsqueeze(1))
        mean = torch.mean(exp_tensor, dim=1, keepdim=True)
        std = torch.std(exp_tensor,dim=1, keepdim=True).clamp(1e-9)
        normalized_tensor = ( exp_tensor - mean ) /std
        return normalized_tensor
    
    def forward_normalized(self, source, target):
        ncc_score = torch.mean(torch.sum(source * target, dim=1, keepdim=True), dim=2)
        return ncc_score
    
    def forward(self, source, target):
        source = self.normalize(source)
        target = self.normalize(target)
        ncc_score = self.forward_normalized(source, target)
        return ncc_score

    def generate_matched_target(self, source, target):
        ncc_score = self.forward(source, target)
        idx = torch.max(ncc_score, dim=0, keepdim=True)[1].repeat(1,3,1,1)
        reference = torch.gather(target, 0, idx)
        return reference



class PatchDistance(nn.Module):
    def __init__(self, patch_size):
        super(PatchDistance, self).__init__()
        expanding_kernel = torch.zeros(3*patch_size*patch_size,1,3,patch_size, patch_size)
        cnt = 0
        for i in range(3):
            for j in range(patch_size):
                for k in range(patch_size):     
                    expanding_kernel[cnt,0,i,j,k] = 1
                    cnt += 1
        
        self.expanding_layer = nn.Conv3d(1, 3*patch_size*patch_size, (3,patch_size,patch_size), padding=(0,patch_size//2,patch_size//2), bias=False)
        self.expanding_layer.weight.data = expanding_kernel
        
    def expanding(self, tensor):
        exp_tensor = self.expanding_layer(tensor.unsqueeze(1))
        return exp_tensor
    
    def forward(self, source, target):
        source = self.expanding(source)
        target = self.expanding(target)
        diff   = torch.mean(torch.abs(source - target), dim=[1,2]).unsqueeze(1)
        return diff

    def generate_matched_target(self, source, target):
        diff = self.forward(source, target)
        idx = torch.min(diff, dim=0, keepdim=True)[1].repeat(1,3,1,1)
        reference = torch.gather(target, 0, idx)
        return reference

def generate_gaussian_kernel(kernel_size): 
    X, Y = torch.meshgrid(torch.linspace(-kernel_size,kernel_size,kernel_size), torch.linspace(-kernel_size,kernel_size,kernel_size))
    kernel = torch.exp(-torch.sqrt(X**2.0 + Y**2.0))
    kernel = kernel / kernel.sum()
    return kernel 



def generate_uniform_kernel(kernel_size): 
    kernel = torch.ones(kernel_size,kernel_size)
    kernel = kernel / kernel.sum()
    return kernel 



def criterion_matching(source, target, kernel_size=15):     
    assert isinstance(source, torch.Tensor)
    assert (len(source) == 1) or (len(target) == 1)
    diff = torch.mean(torch.abs(source-target), dim=1, keepdim=True)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size).float().view(1,1,kernel_size,kernel_size).cuda()
    
    diff = F.conv2d(diff, kernel)
    index = torch.min(diff, dim=0, keepdim=True)[1]
    if len(source)==1:
        target_squeezed = torch.gather(target, 0, index.repeat(1,source.shape[1],1,1))
        return F.mse_loss(source, target_squeezed)
    else:
        source_squeezed = torch.gather(source, 0, index.repeat(1,source.shape[1],1,1))
        return F.mse_loss(source_squeezed, target)



def criterion_matching_cycle(source, target, reflected_xyz, kernel_size=15):
    assert isinstance(source, torch.Tensor)
    assert (len(source) == 1) or (len(target) == 1)
    kernel = generate_uniform_kernel(kernel_size).float().view(1,1,kernel_size,kernel_size).cuda()
    diff = torch.mean(torch.abs(source-target), dim=1, keepdim=True)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    diff = F.conv2d(diff, kernel)
    
    index = torch.min(diff, dim=0, keepdim=True)[1]
    reflected_refer_grad_seq = gatherGlassColorbyXYZ_rev(reflected_refer_xyz_seq, refer_grad)

    target_squeezed = torch.gather(target, 0, index.repeat(1,source.shape[1],1,1))
    return F.mse_loss(source, target_squeezed)
    

def criterion_matching_contrastive(source, target):     
    assert isinstance(source, torch.Tensor)
    assert (len(source) == 1) or (len(target) == 1)
    diff = torch.mean(torch.abs(source-target), dim=1, keepdim=True)
    kernel_size = 15
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()
    diff = F.conv2d(diff, kernel)
    index = torch.min(diff, dim=0, keepdim=True)[1]
    
    if len(source)==1:
        target_squeezed = torch.gather(target, 0, index.repeat(1,source.shape[1],1,1))
        matching_loss = F.mse_loss(source, target_squeezed)
        mask = torch.ones_like(target)
        mask[index] = 0
        mismatching_loss = torch.sum(source * target * mask.detach(), dim=1).mean()
        
        return matching_loss, mismatching_loss
    else:
        source_squeezed = torch.gather(source, 0, index.repeat(1,source.shape[1],1,1))
        matching_loss = F.mse_loss(source_squeezed, target)
        mask = torch.ones_like(source)
        mask[index] = 0
        mismatching_loss = torch.sum(source * target * mask.detach(), dim=1).mean()
        
        return matching_loss, mismatching_loss



def criterion_matching_with_glass(source, target, glass):     
    assert isinstance(source, torch.Tensor)
    assert (len(source) == 1) or (len(target) == 1)
    diff = torch.mean(torch.abs(glass-target), dim=1, keepdim=True)
    kernel_size = 15
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()
    diff = F.conv2d(diff, kernel)
    index = torch.min(diff, dim=0, keepdim=True)[1]
    if len(source)==1:
        target_squeezed = torch.gather(target, 0, index.repeat(1,source.shape[1],1,1))
        return F.mse_loss(source, target_squeezed)
    else:
        source_squeezed = torch.gather(source, 0, index.repeat(1,source.shape[1],1,1))
        return F.mse_loss(source_squeezed, target)


def criterion_matching_with_glass_contrastive(source, target, glass):     
    assert isinstance(source, torch.Tensor)
    assert (len(source) == 1) or (len(target) == 1)
    diff = torch.mean(torch.abs(glass-target), dim=1, keepdim=True)
    kernel_size = 15
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()
    diff = F.conv2d(diff, kernel)
    index = torch.min(diff, dim=0, keepdim=True)[1]
    if len(source)==1:
        target_squeezed = torch.gather(target, 0, index.repeat(1,source.shape[1],1,1))
        matching_loss = F.mse_loss(source, target_squeezed)
        mask = torch.ones_like(target)
        mask[index] = 0
        mismatching_loss = torch.sum(source * target * mask.detach(), dim=1).mean()
        
        return matching_loss, mismatching_loss
    else:
        source_squeezed = torch.gather(source, 0, index.repeat(1,source.shape[1],1,1))
        matching_loss = F.mse_loss(source_squeezed, target)
        mask = torch.ones_like(source)
        mask[index] = 0
        mismatching_loss = torch.sum(torch.abs(source) * torch.abs(target) * mask.detach(), dim=1).mean()
        
        return matching_loss, mismatching_loss



def matching_refer_roll(refle, ReferSeq, roll=5):     
    device = refle.device
    ReferSeq_cpu = ReferSeq.detach().cpu()
    refle_cpu = refle.detach().cpu()
    nb, nc, h, w = ReferSeq.shape
    diff_global = torch.ones(1, 1, 512, 512)
    Refer_cpu_global = torch.ones(1,nc,512, 512) * 99.0
    for i in range(-roll, roll+1):
        for j in range(-roll, roll+1):
            ReferSeq_roll_cpu = torch.roll(ReferSeq_cpu, (i,j), (2,3))
            diff = torch.mean(torch.abs(refle_cpu.expand(nb, -1, -1, -1)-ReferSeq_roll_cpu), dim=1, keepdim=True)
            kernel_size = 15
            diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
            # kernel = generate_gaussian_kernel(kernel_size)
            kernel = generate_uniform_kernel(kernel_size)
            kernel = kernel.float().view(1,1,kernel_size,kernel_size).to(diff.device)
            diff = F.conv2d(diff, kernel)
            diff_min, index = torch.min(diff, dim=0, keepdim=True)
            Refer_cpu = torch.gather(ReferSeq_cpu, 0, index.repeat(1,nc,1,1))
            mask = diff_global.repeat(1,nc,1,1) > diff_min.repeat(1,nc,1,1)
            Refer_cpu_global[mask] = Refer_cpu[mask] 
    Refer = Refer_cpu_global.to(device)

    return Refer
    

def matching_refer(source, targets, source_index=-1):     
    device = source.device
    targets_cpu = targets.detach().cpu()
    source_cpu = source.detach().cpu()
    nb, nc, h, w = targets.shape
    
    kernel_size = 5
    diff = torch.mean(torch.abs(source_cpu.expand(nb, -1, -1, -1)-targets_cpu), dim=1, keepdim=True)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).to(diff.device)
    diff = F.conv2d(diff, kernel)
    index = torch.min(diff, dim=0, keepdim=True)[1]
    target_cpu = torch.gather(targets_cpu, 0, index.repeat(1,nc,1,1))
    target = target_cpu.to(device)
    return target

def matching_refer_cycle(sources, targets, source_index=-1):     
    device = sources.device
    sources_cpu = sources.detach().cpu()
    targets_cpu = targets.detach().cpu()
    nb, nc, h, w = targets.shape
    
    kernel_size = 5
    diff = torch.mean(torch.abs(sources_cpu.unsqueeze(1)-targets_cpu.unsqueeze(0)), dim=2, keepdim=True)
    prob = F.softmax(-1*diff, dim=0) + F.softmax(-1*diff, dim=1)
    prob = prob[source_index]
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).to(diff.device)
    # prob = F.pad(prob, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # prob = F.conv2d(prob, kernel)
    index = torch.max(prob, dim=0, keepdim=True)[1]
    target_cpu = torch.gather(targets_cpu, 0, index.repeat(1,nc,1,1))
    target = target_cpu.to(device)
    return target

# def matching_refer_cycle(sources, targets, source_index=-1):     
#     target = matching_refer(sources[[source_index]], targets)
#     source = matching_refer(target, sources)
#     mask = (source == sources[[source_index]]).float()
#     return target, mask

def criterion_grad_sim_neighbor(refle_grad, ReferSeq_grad, Glass_grad):     
    if isinstance(refle_grad, list):
        loss = 0.0
        for k in range(len(refle_grad)):
            loss = loss + criterion_grad_sim_neighbor(refle_grad[k], ReferSeq_grad[k], Glass_grad[k])
        return loss / float(len(refle_grad))
    else:
        if len(ReferSeq_grad) > 1:     
            sim = torch.mean(torch.abs((Glass_grad*4.0).clamp(-1,1).expand(len(ReferSeq_grad), -1, -1, -1)*ReferSeq_grad), dim=1, keepdim=True)
            
            kernel_size = 15
            sim = F.pad(sim, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
            # kernel = generate_gaussian_kernel(kernel_size)
            kernel = generate_uniform_kernel(kernel_size)
            kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()
            

            sim = F.conv2d(sim, kernel)
            index = torch.max(sim, dim=0, keepdim=True)[1]
            Refer_grad = torch.gather(ReferSeq_grad, 0, index.repeat(1,refle_grad.shape[1],1,1))
        else:
            Refer_grad = ReferSeq
        return F.mse_loss(refle_grad, Refer_grad)


def criterion_refer_neighbor(refle, ReferSeq, Glass):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_refer_neighbor(refle[k], ReferSeq[k], Glass[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(Glass.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            # diff = F.pad(diff, (1,1,1,1), 'replicate')
            # kernel = torch.tensor([[1,2,1,],[2,4,2],[1,2,1]]).float().view(1,1,3,3).cuda()
            # diff = F.pad(diff, (2,2,2,2), 'replicate')
            # kernel = torch.tensor([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]).float().view(1,1,5,5).cuda()
            # kernel = kernel / kernel.sum()
            
            kernel_size = 15
            diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
            # kernel = generate_gaussian_kernel(kernel_size)
            kernel = generate_uniform_kernel(kernel_size)
            kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()

            diff = F.conv2d(diff, kernel)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return F.mse_loss(refle, Refer)
        # channel_weight = torch.tensor([1,1,2]).float().to(refle.device).view(1,3,1,1)
        # if refle.shape[1] == 6:
        #     channel_weight= channel_weight.repeat(1,2,1,1)
        # return F.mse_loss(refle*channel_weight, Refer*channel_weight)


def criterion_refer_neighbor_shift(refle, ReferSeq, Glass, shift=2):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_refer_neighbor(refle[k], ReferSeq[k], Glass[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            Target = Glass.clone()
            diff_min_global = torch.ones(1, 1, Glass.shape[2], Glass.shape[3]).to(Glass.device)
            X, Y = torch.meshgrid(torch.linspace(-1, 1, Glass.shape[3]), torch.linspace(-1, 1, Glass.shape[2]))
            Y, X = Y.unsqueeze(2).unsqueeze(0).to(Glass.device), X.unsqueeze(2).unsqueeze(0).to(Glass.device)
            
            for i in range(-shift, shift):
                for j in range(-shift, shift):
                    X_ = X + float(i) / float(Glass.shape[3])
                    Y_ = Y + float(j) / float(Glass.shape[2])
                    grid = torch.cat((Y_, X_), 3)
                    Glass_tmp = F.grid_sample(Glass, grid, align_corners=False, padding_mode='reflection')
                    # ReferSeq = F.grid_sample(ReferSeq, grid, align_corners=False, padding_mode='replicate')
                    diff = torch.mean(torch.abs(Glass_tmp.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
                    kernel_size = 15
                    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
                    # kernel = generate_gaussian_kernel(kernel_size)
                    kernel = generate_uniform_kernel(kernel_size)
                    kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()

                    diff = F.conv2d(diff, kernel)
                    diff_min, index = torch.min(diff, dim=0, keepdim=True)
                    Target_tmp = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
                    Target[diff_min < diff_min_global] = Target_tmp[diff_min < diff_min_global] 
            
        else:
            Target = ReferSeq
        return F.mse_loss(refle, Target)
        


def epipolar_grids(h, w, num=60, min_value=0.1, max_value=1.5):
    
    X, Y = torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h))
    grid = torch.cat((Y.unsqueeze(2), X.unsqueeze(2)), 2)
    grid = grid.unsqueeze(0)
    grid_list = []
    grid_inv_list = []
    for k in torch.linspace(min_value, max_value, num):
        scale = float(k.item())
        if scale < 0.1:
            break
        grid_list.append(grid * scale)
        grid_inv_list.append(grid / scale)
    grids = torch.cat(grid_list, 0)
    grids_inv = torch.cat(grid_inv_list, 0)
    return grids, grids_inv

def criterion_refer_neighbor_cycle(refle, Refer, Glass, Grids, GridsInv):     
    assert len(Refer) > 1, "Single reference"
    ReferSeq = F.grid_sample(Refer, Grids, align_corners=False)
    kernel_size = 7
    torch.mean(torch.abs(Glass.unsqueeze(1) - ReferSeq.unsqueeze(0)), dim=2, keepdim=True)
    diff = torch.mean(torch.abs(Glass.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    # kernel = generate_gaussian_kernel(kernel_size)
    kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,kernel_size,kernel_size).cuda()
    diff = F.conv2d(diff, kernel)


    index = torch.min(diff, dim=0, keepdim=True)[1]
    
    Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
    return F.mse_loss(refle, Refer)




def cycle_matching(GlassSeq, ReferSeq):     
    
    kernel_size = 51
    kernel = generate_gaussian_kernel(kernel_size)
    # kernel = generate_uniform_kernel(kernel_size)
    kernel = kernel.float().view(1,1,1,kernel_size,kernel_size).to(GlassSeq.device)
    
    Glass = GlassSeq[[-1]]
    # torchvision.utils.save_image(Glass[:,:3,:,:], 'tmp.png')

    diff = torch.mean(torch.abs(Glass.unsqueeze(1) - ReferSeq.unsqueeze(0)), dim=2)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    diff = F.conv3d(diff.unsqueeze(1), kernel).squeeze(1)

    diff_sort_arg = torch.argsort(diff, dim=1, descending=False)
    diff_sort_arg = diff_sort_arg[:,:1,:,:]
    diff_sort_arg = torch.split(diff_sort_arg, 1, 1)
    Refer_list = []
    for k in range(len(diff_sort_arg)):
        Refer_list.append(torch.gather(ReferSeq, 0, diff_sort_arg[k].repeat(1,ReferSeq.shape[1],1,1)))

    Refer = torch.cat(Refer_list, 0)

    diff = torch.mean(torch.abs(GlassSeq.unsqueeze(1) - Refer.unsqueeze(0)), dim=2)
    diff = F.pad(diff, (kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2), 'replicate')
    diff = F.conv3d(diff.unsqueeze(1), kernel).squeeze(1)

    diff_sort_arg = torch.argsort(diff, dim=0, descending=False)
    weight = (diff_sort_arg[:5,:,:,:] == (len(GlassSeq)-1)).float().sum(dim=0, keepdim=True).clamp(max=1)
    
    
    return Refer, weight


def criterion_refer(refle, ReferSeq, Glass):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_refer(refle[k], ReferSeq[k], Glass[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(Glass.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return F.mse_loss(refle, Refer)

def criterion_statics(refle, Refer, std_weight=1.0):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_statics(refle[k], Refer[k], std_weight)
        return loss / float(len(refle))
    else:
        mean_diff = F.mse_loss(refle.mean(dim=[0, 2,3]), Refer.mean(dim=[0, 2,3]))
        std_diff = F.mse_loss(refle.std(dim=[0, 2,3]), Refer.std(dim=[0, 2,3]))
        return mean_diff + std_weight * std_diff


def criterion_mse(refle, ReferSeq):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_mse(refle[k], ReferSeq[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(refle.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return F.mse_loss(refle, Refer)



def criterion_mse_multiscale(refle, ReferSeq):     
    if isinstance(refle, list):
        loss = 0.0
        weight = 1.0
        for k in range(len(refle)):
            loss = loss + weight *  criterion_mse_multiscale(refle[k], ReferSeq[k])
            weight = weight / 2.0
        return loss
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(refle.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return F.mse_loss(refle, Refer)


def criterion_mse_weight(refle, TargetRefer, weight):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_mse_weight(refle[k], TargetRefer[k], weight[k])
        return loss / float(len(refle))
    else:
        return (F.mse_loss(refle, TargetRefer, reduction='none') * weight).mean()



def criterion_mse_blue(refle, ReferSeq):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_mse_blue(refle[k], ReferSeq[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(refle.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        
        channel_weight = torch.tensor([1,1,4]).float().to(refle.device).view(1,3,1,1)
        if refle.shape[1] == 6:
            channel_weight= channel_weight.repeat(1,2,1,1)
        return F.mse_loss(refle*channel_weight, Refer*channel_weight)



def criterion_clip_sparsity(refle_grad, epsilon=(0.1,0.1,0.1)):     
    if isinstance(refle_grad, list):
        loss = 0.0
        for k in range(len(refle_grad)):
            loss = loss + criterion_clip_sparsity(refle_grad[k],  epsilon)
        return loss / float(len(refle_grad))
    else:
        refle_grad_channels = torch.split(refle_grad, 1,1)
        loss = [] 

        for k in range(len(refle_grad_channels)):
            
            loss.append(F.mse_loss(refle_grad_channels[k].clamp(min=-epsilon[k%3], max=epsilon[k%3]), \
                        torch.zeros_like(refle_grad_channels[k]), reduction='mean'))

        return sum(loss) / float(len(loss))


def criterion_clip_sparsity_weight(refle_grad,  weight, epsilon=(0.1,0.1,0.1)):     
    if isinstance(refle_grad, list):
        loss = 0.0
        for k in range(len(refle_grad)):
            loss = loss + criterion_clip_sparsity(refle_grad[k], weight[k], epsilon)
        return loss / float(len(refle_grad))
    else:
        refle_grad_channels = torch.split(refle_grad, 1,1)
        loss = [] 

        for k in range(len(refle_grad_channels)):
            loss_map = F.mse_loss(refle_grad_channels[k].clamp(min=-epsilon[k//3], max=epsilon[k//3]), \
                        torch.zeros_like(refle_grad_channels[k]), reduction='none')
            loss.append((loss_map * weight).mean())

        return sum(loss) / float(len(loss))

def criterion_mse_with_weight(refle, ReferSeq, weight):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_mse_with_weight(refle[k], ReferSeq[k], weight[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(refle.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return (F.mse_loss(refle, Refer, reduction='none') * weight).sum() / weight.sum()


def criterion_l1(refle, ReferSeq):     
    if isinstance(refle, list):
        loss = 0.0
        for k in range(len(refle)):
            loss = loss + criterion_l1(refle[k], ReferSeq[k])
        return loss / float(len(refle))
    else:
        if len(ReferSeq) > 1:     
            diff = torch.mean(torch.abs(refle.expand(len(ReferSeq), -1, -1, -1)-ReferSeq), dim=1, keepdim=True)
            index = torch.min(diff, dim=0, keepdim=True)[1]
            Refer = torch.gather(ReferSeq, 0, index.repeat(1,refle.shape[1],1,1))
        else:
            Refer = ReferSeq
        return F.l1_loss(refle, Refer)

def criterion_overlap(trans_grad, refle_grad):
    if isinstance(trans_grad, list):
        loss = 0.0 
        for k in range(len(refle_grad)):
            loss = loss + criterion_overlap(trans_grad[k], refle_grad[k])
        return loss / float(len(refle_grad))
    else:
        edge1 = torch.max(torch.abs(trans_grad), dim=1)[0]
        edge2 = torch.max(torch.abs(refle_grad), dim=1)[0]
        return F.mse_loss(edge1*edge2, torch.zeros_like(edge2))


def criterion_overlap2(trans_grad_mag, refle_grad_mag):
    if isinstance(trans_grad_mag, list):
        loss = 0.0 
        for k in range(len(refle_grad_mag)):
            loss = loss + criterion_overlap2(trans_grad_mag[k], refle_grad_mag[k])
        return loss / float(len(refle_grad_mag))
    else:
        return F.mse_loss(trans_grad_mag*refle_grad_mag, torch.zeros_like(refle_grad_mag))


def criterion_overlap3(trans_grad_mag, refle_grad_mag):
    if isinstance(trans_grad_mag, list):
        loss = 0.0 
        for k in range(len(refle_grad_mag)):
            loss = loss + criterion_overlap3(trans_grad_mag[k], refle_grad_mag[k])
        return loss / float(len(refle_grad_mag))
    else:
        return torch.sum(trans_grad_mag*refle_grad_mag, dim=1).mean()



def criterion_overlap_multiscale(trans_grad, refle_grad):
    if isinstance(trans_grad, list):
        loss = 0.0 
        weight = 1.0
        for k in range(len(refle_grad)):
            loss = loss  +  weight * criterion_overlap_multiscale(trans_grad[k], refle_grad[k])
            weight = weight / 2.0
        return loss / float(len(refle_grad))
    else:
        return F.mse_loss(torch.abs(trans_grad)*torch.abs(refle_grad), torch.zeros_like(refle_grad))



# def criterion_overlap2(trans_grad, refle_grad):
#     if isinstance(trans_grad, list):
#         loss = 0.0 
#         for k in range(len(refle_grad)):
#             loss = loss + criterion_overlap(trans_grad[k], refle_grad[k])
#         return loss / float(len(refle_grad))
#     else:
#         trans_grad_norm = torch.norm(trans_grad, dim=[1,2,3],keepdim=True)
#         refle_grad_norm = torch.norm(refle_grad, dim=[1,2,3],keepdim=True)
#         lambda_t = torch.sqrt(refle_grad_norm / trans_grad_norm)
#         lambda_r = torch.sqrt(trans_grad_norm / refle_grad_norm)
#         excl = torch.tanh(lambda_t * torch.abs(trans_grad)) * torch.tanh(lambda_r * torch.abs(refle_grad)) 
#         return excl.norm()



def criterion_grad_invalidity(trans_grad, Glass_grad, eps=1e-4, threshold=1.0):
    if isinstance(trans_grad, list):
        loss = 0.0 
        for k in range(len(trans_grad)):
            loss = loss + criterion_grad_invalidity(trans_grad[k], Glass_grad[k], eps, threshold)
        return loss / float(len(trans_grad))
    else:
        return F.mse_loss((torch.abs(trans_grad)/(torch.abs(Glass_grad)+eps)).clamp(threshold), torch.ones_like(trans_grad))

def criterion_adv(fake, target=True):
    if isinstance(fake, list):
        loss = 0.0 
        for k in range(len(fake)):
            loss = loss + criterion_adv(fake[k], target)
        return loss / float(len(fake))
    else:
        if target:
            return F.mse_loss(fake, torch.ones_like(fake))
        else:
            return F.mse_loss(fake, torch.zeros_like(fake))
        
def criterion_saturation(trans, Glass):
    if isinstance(trans, list):
        loss = 0.0 
        for k in range(len(trans)):
            loss = loss + criterion_saturation(trans[k], Glass[k])
        return loss / float(len(trans))
    else:
        return F.mse_loss(F.relu(trans-Glass), torch.zeros_like(trans))         

class Blur(nn.Module):
    def __init__(self, sigma):
        super(Blur, self).__init__()
        
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
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
        
        paddingl = (kernel_size - 1) // 2
        paddingr = kernel_size - 1 - paddingl
        pad = torch.nn.ReplicationPad3d((paddingl, paddingr, paddingl, paddingr, 0, 0))
        gaussian_filter = nn.Conv3d(in_channels=1, out_channels=1,
                                    kernel_size=(1,kernel_size,kernel_size), bias=False)
                                    
        
        gaussian_filter.weight.data = gaussian_kernel.cuda()
        gaussian_filter.weight.requires_grad = False
        
        self.blur_layer = nn.Sequential(
            pad, 
            gaussian_filter,
        )
    def forward(self, x):
        
        x = self.blur_layer(x.unsqueeze(1)).squeeze(1)

        return x


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = -grad_outputs
        return grad_inputs

class GSL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = 0.00001 * grad_outputs
        return grad_inputs



class GSL10(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = 10* grad_outputs
        return grad_inputs



class GSL200(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = 200* grad_outputs
        return grad_inputs


class GradientClipper3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        if torch.abs(grad_outputs).max() > 1e-0:
            grad_inputs = grad_outputs / torch.abs(grad_outputs).max() * 1e-0
        else:
            grad_inputs = grad_outputs
        return grad_inputs

# class Encoder(nn.Module):
#     """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
#     def __init__(self, img, patch_size):
#         """Construct a 1x1 PatchGAN discriminator
#         """
#         super(Encoder, self).__init__()

#         _, _, h, w = img.shape
#         patches = []
#         for k1 in range(h // patch_size):
#             for k2 in range(w // patch_size):
#                 if ((k1+1)*patch_size<(h+1)) and ((k2+1)*patch_size<(w+1)):
#                     patches.append(img[:, :, k1*patch_size:(k1+1)*patch_size, k2*patch_size:(k2+1)*patch_size])
        
#         img2 = F.interpolate(img, (int(img.shape[2]*0.7), int(img.shape[3]*0.7)), mode='bilinear', align_corners=False)
#         _, _, h, w = img2.shape
#         for k1 in range(h // patch_size):
#             for k2 in range(w // patch_size):
#                 if ((k1+1)*patch_size<(h+1)) and ((k2+1)*patch_size<(w+1)):
#                     patches.append(img2[:, :, k1*patch_size:(k1+1)*patch_size, k2*patch_size:(k2+1)*patch_size])
        
#         self.kernels = torch.cat(patches,0).detach().clone().cuda()
#         self.ndf = self.kernels.shape[0]
#         self.patch_size = self.kernels.shape[3]
        
#         self.pad = nn.ReplicationPad2d(self.patch_size//2)
#         self.nfeats = self.kernels.shape[0]

#     def forward(self, input):
#         """Standard forward."""
#         # input = self.pad(input)
#         act = F.conv2d(input, self.kernels)
        
#         # act = self.encoder(input)
#         return act

class Encoder(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, output_nc, ndf):
        """Construct a 1x1 PatchGAN discriminator
        """
        super(Encoder, self).__init__()
        self.ndf = ndf
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=7, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=7, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=7, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 2, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.ndf, self.ndf*2, kernel_size=3, stride=1, padding=0, bias=True, dilation=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.ndf*2, self.ndf, kernel_size=3, stride=1, padding=0, bias=True, dilation=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.ndf, 8, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.ReLU(),
            
        )
        self.encoder[0].weight.data.div(self.ndf*4)
        self.encoder[2].weight.data.div(self.ndf*4)
        self.encoder[4].weight.data.div(self.ndf*4)
        # self.encoder[4].weight.data.div(ndf)
        # self.encoder[6].weight.data.div(ndf)
        
    def forward(self, input):
        """Standard forward."""
        act = self.encoder(input)
        return act


class Decoder(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, img, patch_size):
        """Construct a 1x1 PatchGAN discriminator"""

        super(Decoder, self).__init__()

        
        _, _, h, w = img.shape
        patches = []
        for k1 in range(h // patch_size):
            for k2 in range(w // patch_size):
                if ((k1+1)*patch_size<(h+1)) and ((k2+1)*patch_size<(w+1)):
                    patches.append(img[:, :, k1*patch_size:(k1+1)*patch_size, k2*patch_size:(k2+1)*patch_size])
        
        img2 = F.interpolate(img, (int(img.shape[2]*0.7), int(img.shape[3]*0.7)), mode='bilinear', align_corners=False)
        _, _, h, w = img2.shape
        for k1 in range(h // patch_size):
            for k2 in range(w // patch_size):
                if ((k1+1)*patch_size<(h+1)) and ((k2+1)*patch_size<(w+1)):
                    patches.append(img2[:, :, k1*patch_size:(k1+1)*patch_size, k2*patch_size:(k2+1)*patch_size])
        
        self.kernels = torch.cat(patches,0).detach().clone().cuda()
        self.ndf = self.kernels.shape[0]
        self.patch_size = self.kernels.shape[3]
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3,   kernel_size=3, stride=1, padding=0, bias=True),
        )
        self.decoder[0].weight.data.div(self.ndf*4)
        self.decoder[2].weight.data.div(self.ndf*4)
        
    def forward(self, feats):
        """Standard forward."""
        act = F.conv2d(feats, self.kernels, stride=1, padding=0, bias=None)
        output = self.decoder(act)
        return output


class Classifier(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, ndf=64, num_stripes=4):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Classifier, self).__init__()
        self.ndf = ndf
        self.num_classes = num_stripes+1
        self.num_stripes = num_stripes
        self.decoder = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.num_classes,   kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.decoder[0].weight.data.div(self.ndf)
        self.decoder[2].weight.data.div(self.ndf)
        self.decoder[4].weight.data.div(self.ndf)
         
    def forward(self, img):
        """Standard forward."""
        scores = self.decoder(img)
        patch_height = scores.shape[2]//self.num_stripes
        width = scores.shape[3]
        scores = F.interpolate(scores, ((patch_height)*self.num_stripes,width), mode='bilinear', align_corners=False)
        scores = F.avg_pool2d(scores, (patch_height, width))
        return scores

    def predict(self,img):
        scores = self.decoder(img)
        return scores

class Generator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, image_size=(512,512), ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Generator, self).__init__()
        self.ndf = ndf
        self.decoder = nn.Sequential(
            nn.Conv2d(4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 3,   kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.decoder[0].weight.data.div(self.ndf)
        self.decoder[2].weight.data.div(self.ndf)
        self.decoder[4].weight.data.div(self.ndf)
        self.image_size = image_size
        input = torch.zeros(image_size).unsqueeze(0).unsqueeze(0)
        self.input = nn.Parameter(input)
    def forward(self, img):
        """Standard forward."""
        noise = F.interpolate(self.input, img.shape[-2], mode='bilinear', align_corners=False)
        x = torch.cat((img,noise), dim=1)
        output = img + self.decoder(x)
        return output

class AutoEncoder(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 2, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        self.auto_encoder[0].weight.data.div(self.ndf*4)
        self.auto_encoder[2].weight.data.div(self.ndf*4)
        self.auto_encoder[4].weight.data.div(self.ndf*4)
        self.auto_encoder[6].weight.data.div(self.ndf*4)
        self.auto_encoder[8].weight.data.div(self.ndf*4)
        
        self.auto_encoder2 = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 2, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        self.auto_encoder2[0].weight.data.div(self.ndf*4)
        self.auto_encoder2[2].weight.data.div(self.ndf*4)
        self.auto_encoder2[4].weight.data.div(self.ndf*4)
        self.auto_encoder2[6].weight.data.div(self.ndf*4)
        self.auto_encoder2[8].weight.data.div(self.ndf*4)

    def forward_part(self, input):
        """Standard forward."""
        scores = self.auto_encoder(input)
        mask = F.softmax(scores, 1)[:,[1],:,:]
        output = input * mask
        return output, scores
        
    
    def forward(self, input):
        """Standard forward."""
        input_up, input_down = torch.split(input, [input.shape[2]//2, input.shape[2] - input.shape[2]//2], 2)
        output_up, score_up = self.forward_part(input_up)
        output_down, score_down = self.forward_part(input_down)
        output = torch.cat((output_up, output_down), 2)
        scores = torch.cat((score_up, score_down), 2)
        return output, scores



class AutoEncoder2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder2, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(5, self.ndf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.auto_encoder[0].weight.data.div(self.ndf*4)
        self.auto_encoder[2].weight.data.div(self.ndf*4)
        self.auto_encoder[4].weight.data.div(self.ndf*4)
        # self.auto_encoder[6].weight.data.div(self.ndf*4)
        # self.auto_encoder[8].weight.data.div(self.ndf*4)
        
        self.auto_encoder2 = nn.Sequential(
            nn.Conv2d(5, self.ndf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.auto_encoder2[0].weight.data.div(self.ndf*4)
        self.auto_encoder2[2].weight.data.div(self.ndf*4)
        self.auto_encoder2[4].weight.data.div(self.ndf*4)
        # self.auto_encoder2[6].weight.data.div(self.ndf*4)
        # self.auto_encoder2[8].weight.data.div(self.ndf*4)

    def forward_part(self, input, up=True):
        """Standard forward."""
        
        Y = torch.linspace(0, 1, input.shape[2]).view(1,1,-1,1).repeat(input.shape[0],1,1,input.shape[3]).cuda()
        X = torch.linspace(0, 1, input.shape[3]).view(1,1,1,-1).repeat(input.shape[0],1,input.shape[2],1).cuda()
        input2 = torch.cat((input,X, Y), 1)
        if up :
            output = input * self.auto_encoder(input2)
        else: 
            output = input * self.auto_encoder2(input2)
        return output
        
    # def forward(self, input, switch=False):
    #     """Standard forward."""
    #     input_up, input_down = torch.split(input, [input.shape[2]//2, input.shape[2] - input.shape[2]//2], 2)
    #     if not switch:
    #         output_up = self.forward_part(input_up, up=True)
    #         output_down = self.forward_part(input_down, up=False)
    #         output = torch.cat((output_up, output_down), 2)
    #     else:
    #         output_up = self.forward_part(input_up, up=False)
    #         output_down = self.forward_part(input_down, up=True)
    #         output = torch.cat((output_up, output_down), 2)
    #     return output
    
    def forward(self, input, switch=False):
        """Standard forward."""
        output = self.forward_part(input)
        return output

        # input_up, input_down = torch.split(input, [input.shape[2]//2, input.shape[2] - input.shape[2]//2], 2)
        # if not switch:
        #     output_up = self.forward_part(input_up, up=True)
        #     output_down = self.forward_part(input_down, up=False)
        #     output = torch.cat((output_up, output_down), 2)
        # else:
        #     output_up = self.forward_part(input_up, up=False)
        #     output_down = self.forward_part(input_down, up=True)
        #     output = torch.cat((output_up, output_down), 2)
        # return output


class AutoEncoder3(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder3, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(11, self.ndf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        self.auto_encoder[0].weight.data.div(self.ndf*4)
        self.auto_encoder[2].weight.data.div(self.ndf*4)
        self.auto_encoder[4].weight.data.div(self.ndf*4)
        
        self.auto_encoder2 = nn.Sequential(
            nn.Conv2d(11, self.ndf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        self.auto_encoder2[0].weight.data.div(self.ndf*4)
        self.auto_encoder2[2].weight.data.div(self.ndf*4)
        self.auto_encoder2[4].weight.data.div(self.ndf*4)
        self.color_grad_layer = GradientLayer()



    
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

    def gradient(self, x):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel).squeeze(1)
        
        dx = self.g_clip(dx)
        dy = self.g_clip(dy)
        return dx, dy

    def forward_part(self, input, up=True):
        """Standard forward."""
        
        Y = torch.linspace(0, 1, input.shape[2]).view(1,1,-1,1).repeat(input.shape[0],1,1,input.shape[3]).cuda()
        X = torch.linspace(0, 1, input.shape[3]).view(1,1,1,-1).repeat(input.shape[0],1,input.shape[2],1).cuda()
        dx, dy = self.gradient(input)
    
        input2 = torch.cat((input,dx, dy ,X, Y), 1)
        
        if up :
            output = self.auto_encoder(input2)
        else: 
            output = self.auto_encoder2(input2)
        return output
        
    
    def forward(self, input, switch=False):
        """Standard forward."""
        output = self.forward_part(input)
        return output


class AutoEncoder4(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder4, self).__init__()
        self.ndf = ndf
        self.auto_encoder1 = nn.Sequential(
            nn.Conv2d(9, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # self.auto_encoder[0].weight.data.div(self.ndf*4)
        # self.auto_encoder[3].weight.data.div(self.ndf*4)
        # self.auto_encoder[6].weight.data.div(self.ndf*4)
        
        self.auto_encoder2 = nn.Sequential(
            nn.Conv2d(9, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        # self.auto_encoder2[0].weight.data.div(self.ndf*4)
        # self.auto_encoder2[3].weight.data.div(self.ndf*4)
        # self.auto_encoder2[6].weight.data.div(self.ndf*4)

        self.auto_decoder_layer1 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.auto_decoder_layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.auto_decoder_layer3 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
    
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

    def gradient(self, x):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel).squeeze(1)
        
        dx = self.g_clip(dx)
        dy = self.g_clip(dy)
        return dx, dy

    def decode(self, feat, output_shape):
        y1 = self.auto_decoder_layer1(feat)
        y1 = F.interpolate(y1, (y1.shape[2]*2,y1.shape[3]*2), mode='bilinear', align_corners=False)
        y2 = self.auto_decoder_layer2(y1)
        y2 = F.interpolate(y2, (y2.shape[2]*2,y2.shape[3]*2), mode='bilinear', align_corners=False)
        y3 = self.auto_decoder_layer3(y2)
        output = F.interpolate(y3, output_shape, mode='bilinear', align_corners=False)
        return output


    def decompose(self, input):
        """Standard forward."""
        # Y = torch.linspace(0, 1, input.shape[2]).view(1,1,-1,1).repeat(input.shape[0],1,1,input.shape[3]).cuda()
        # X = torch.linspace(0, 1, input.shape[3]).view(1,1,1,-1).repeat(input.shape[0],1,input.shape[2],1).cuda()
        dx, dy = self.gradient(input)
        input2 = torch.cat((input,dx, dy), 1)

        feat1 = self.auto_encoder1(input2)
        feat2 = self.auto_encoder2(input2)

        trans = self.decode(feat1, input.shape[-2:])
        refle = self.decode(feat2, input.shape[-2:])
        
        return trans, refle

    def reconstruct(self, input):
        # Y = torch.linspace(0, 1, input.shape[2]).view(1,1,-1,1).repeat(input.shape[0],1,1,input.shape[3]).cuda()
        # X = torch.linspace(0, 1, input.shape[3]).view(1,1,1,-1).repeat(input.shape[0],1,input.shape[2],1).cuda()
        dx, dy = self.gradient(input)
        input2 = torch.cat((input,dx, dy), 1)

        feat1 = self.auto_encoder1(input2)
        feat2 = self.auto_encoder2(input2)

        output = self.decode(feat1+feat2, input.shape[-2:])
        return output
    


class AutoEncoder5(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder5, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(6, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        
        self.generator = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))

    def forward(self, source, refer):
        mask = self.auto_encoder(torch.cat((source, refer), 1))
        trans = self.generator(source)
        refle = mask * refer
        return trans, refle
    


class AutoEncoder6(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder6, self).__init__()
        self.ndf = ndf
        self.generator = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))

    def forward(self, refer):
        delta = self.generator(refer)
        refle = delta + refer
        return refle



class AutoEncoder7(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder7, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        # self.trans = nn.Parameter(torch.zeros(1,1,512,512))
    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        refle_feat = self.adain(source_feat, target_feat)
        refle = self.decoder(refle_feat)
        return refle




class AutoEncoder8(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder8, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.decoder_refle = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        self.generate_trans1 = nn.Sequential(
            nn.Conv2d(1, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        self.generate_trans2 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        
        self.generate_trans3 = nn.Sequential(
            nn.Conv2d(self.ndf*2, self.ndf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*2, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        
        self.generate_trans4 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf, 3, kernel_size=1, stride=1, padding=0, bias=False),
        )


        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        self.noise = nn.Parameter(torch.randn(1,1,32, 32))
        self.adain = AdaIN()
        self.trans = nn.Parameter(torch.ones(1,3,512,512))
    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        refle_feat = self.adain(source_feat, target_feat)
        refle = self.decoder_refle(refle_feat)
        # trans_feat = source_feat - refle_feat
        # trans = self.decoder_trans(trans_feat)
        y = self.generate_trans1(self.noise)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.generate_trans2(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.generate_trans3(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.generate_trans4(y)
        trans = F.interpolate(y, (refle.shape[2], refle.shape[3]), mode='bilinear', align_corners=False)
        
        return trans, refle


class AutoEncoder9(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder9, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.masking = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        self.adain2 = AdaIN2()
        self.trans_feat_mean = nn.Parameter(torch.randn(1,self.ndf,1,1))
        self.trans_feat_std = nn.Parameter(torch.abs(torch.randn(1,self.ndf,1,1)))
    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)

        refle_feat = self.adain(source_feat, target_feat)
        trans_feat = self.adain2(source_feat, self.trans_feat_mean, self.trans_feat_std)
        refle = self.decoder(refle_feat)
        trans = self.decoder(trans_feat)

        return trans, refle



class AutoEncoder10(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder10, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))
    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        source_feat_ = torch.sigmoid(self.masking) * source_feat
        refle_feat = self.adain(source_feat_, target_feat)
        refle = self.decoder(refle_feat)

        trans_feat = (1-torch.sigmoid(self.masking)) * source_feat

        # refer_feat = self.encoder(refle)
        trans = self.decoder(trans_feat)
        glass = self.decoder(source_feat)
        return trans, refle, glass


class AutoEncoder11(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder11, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))
    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        source_feat_ = torch.sigmoid(self.masking) * source_feat
        refle_feat = self.adain(source_feat_, target_feat)
        refle = self.decoder(refle_feat)

        trans_feat = (1-torch.sigmoid(self.masking)) * source_feat
        trans_feat_en = self.adain(trans_feat, source_feat)
        # refer_feat = self.encoder(refle)
        trans = self.decoder(trans_feat_en)
        glass = self.decoder(source_feat)
        return trans, refle, glass



class AutoEncoder11(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder11, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))

    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        source_feat_ = torch.sigmoid(self.masking) * source_feat
        refle_feat = self.adain(source_feat_, target_feat)
        refle = self.decoder(refle_feat)

        trans_feat = (1-torch.sigmoid(self.masking)) * source_feat
        trans_feat_en = self.adain(trans_feat, source_feat)
        # refer_feat = self.encoder(refle)
        trans = self.decoder(trans_feat_en)
        glass = self.decoder(source_feat)
        return trans, refle, glass



class AutoEncoder13(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder13, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        # self.noise = nn.Parameter(torch.randn(1,1,512,512))
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))

    
    def return_mask(self):
        return self.masking * 1

    def forward(self, source, target):
        _, _, h, w = source.shape
        source2 = F.interpolate(source, (h//2, w//2), mode='bilinear', align_corners=False)
        target2 = F.interpolate(target, (h//2, w//2), mode='bilinear', align_corners=False)
        masking2 = F.interpolate(self.masking, (h//2, w//2), mode='bilinear', align_corners=False)
        source4 = F.interpolate(source, (h//4, w//4), mode='bilinear', align_corners=False)
        target4 = F.interpolate(target, (h//4, w//4), mode='bilinear', align_corners=False)
        masking4 = F.interpolate(self.masking, (h//4, w//4), mode='bilinear', align_corners=False)

        source_feat4 = self.encoder(source4)
        target_feat4 = self.encoder(target4)
        refle_feat4 = torch.sigmoid(masking4) * source_feat4
        refle_feat4 = self.adain(refle_feat4, target_feat4)
        refle4 = self.decoder(refle_feat4)
        trans_feat4 = (1-torch.sigmoid(masking4)) * source_feat4
        # trans_feat4_en = self.adain(trans_feat4, source_feat4)
        trans4 = self.decoder(trans_feat4)
        glass4 = self.decoder(source_feat4)

        source_feat2 = 0.5*self.encoder(source2) + 0.5*F.interpolate(source_feat4, (h//2, w//2), mode='bilinear', align_corners=False)
        target_feat2 = 0.5*self.encoder(target2) + 0.5*F.interpolate(target_feat4, (h//2, w//2), mode='bilinear', align_corners=False)
        refle_feat2 = torch.sigmoid(masking2) * source_feat2
        refle_feat2 = self.adain(refle_feat2, target_feat2)
        refle2 = self.decoder(refle_feat2)
        trans_feat2 = (1-torch.sigmoid(masking2)) * source_feat2
        # trans_feat2_en = self.adain(trans_feat2, source_feat2)
        trans2 = self.decoder(trans_feat2)
        glass2 = self.decoder(source_feat2)

        source_feat = 0.5*self.encoder(source) + 0.5*F.interpolate(source_feat2, (h, w), mode='bilinear', align_corners=False)
        target_feat = 0.5*self.encoder(target) + 0.5*F.interpolate(source_feat2, (h, w), mode='bilinear', align_corners=False)
        refle_feat = torch.sigmoid(self.masking) * source_feat
        refle_feat = self.adain(refle_feat, target_feat)
        refle = self.decoder(refle_feat)
        trans_feat = (1-torch.sigmoid(self.masking)) * source_feat
        # trans_feat_en = self.adain(trans_feat, source_feat)
        trans = self.decoder(trans_feat)
        glass = self.decoder(source_feat)
        
        return trans4, refle4, glass4, trans2, refle2, glass2, trans, refle, glass




class AutoEncoder14(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoder14, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))

        self.noise = nn.Parameter(torch.randn(1,ndf,1,1))

        self.mean_trans = nn.Sequential(
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
        )
        self.std_trans = nn.Sequential(
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
            nn.ReLU(), 
            nn.Conv2d(ndf, ndf, 1,1,0, bias=False), 
            nn.ReLU(),
        )
    def trans_static(self):
        mean_trans = self.mean_trans(self.noise)
        std_trans = self.std_trans(self.noise)
        return mean_trans, std_trans

    def return_mask(self):
        return self.masking * 1

    def forward(self, source, target):
        _, _, h, w = source.shape
        source2 = F.interpolate(source, (h//2, w//2), mode='bilinear', align_corners=False)
        target2 = F.interpolate(target, (h//2, w//2), mode='bilinear', align_corners=False)
        masking2 = F.interpolate(self.masking, (h//2, w//2), mode='bilinear', align_corners=False)
        source4 = F.interpolate(source, (h//4, w//4), mode='bilinear', align_corners=False)
        target4 = F.interpolate(target, (h//4, w//4), mode='bilinear', align_corners=False)
        masking4 = F.interpolate(self.masking, (h//4, w//4), mode='bilinear', align_corners=False)

        mean_trans, std_trans = self.trans_static()

        source_feat4 = self.encoder(source4)
        target_feat4 = self.encoder(target4)
        refle_feat4 = torch.sigmoid(masking4) * source_feat4
        refle_feat4 = self.adain(refle_feat4, target_feat4)
        refle4 = self.decoder(refle_feat4)
        trans_feat4 = (1-torch.sigmoid(masking4)) * source_feat4
        trans_feat4_en = self.adain.forward2(trans_feat4, mean_trans, std_trans)
        trans4 = self.decoder(trans_feat4_en)
        glass4 = self.decoder(source_feat4)

        source_feat2 = 0.5*self.encoder(source2) + 0.5*F.interpolate(source_feat4, (h//2, w//2), mode='bilinear', align_corners=False)
        target_feat2 = 0.5*self.encoder(target2) + 0.5*F.interpolate(target_feat4, (h//2, w//2), mode='bilinear', align_corners=False)
        refle_feat2 = torch.sigmoid(masking2) * source_feat2
        refle_feat2 = self.adain(refle_feat2, target_feat2)
        refle2 = self.decoder(refle_feat2)
        trans_feat2 = (1-torch.sigmoid(masking2)) * source_feat2
        trans_feat2_en = self.adain(trans_feat2, mean_trans, std_trans)
        trans2 = self.decoder(trans_feat2_en)
        glass2 = self.decoder(source_feat2)

        source_feat = 0.5*self.encoder(source) + 0.5*F.interpolate(source_feat2, (h, w), mode='bilinear', align_corners=False)
        target_feat = 0.5*self.encoder(target) + 0.5*F.interpolate(source_feat2, (h, w), mode='bilinear', align_corners=False)
        refle_feat = torch.sigmoid(self.masking) * source_feat
        refle_feat = self.adain(refle_feat, target_feat)
        refle = self.decoder(refle_feat)
        trans_feat = (1-torch.sigmoid(self.masking)) * source_feat
        trans_feat_en = self.adain(trans_feat, mean_trans, std_trans)
        trans = self.decoder(trans_feat_en)
        glass = self.decoder(source_feat)
        
        return trans4, refle4, glass4, trans2, refle2, glass2, trans, refle, glass


class AutoEncoderSub(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(AutoEncoderSub, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
    def upsample(self, x):
        x = F.interpolate(x, (x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=False)
        return x 
    def forward(self,x):
        y = self.encoder(x)
        y = self.upsample(y)
        y = self.decoder1(y)
        y = self.upsample(y)
        y = self.decoder2(y)
        return y

class Generator2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Generator2, self).__init__()
        self.ndf = ndf
        self.output_size=(512,512)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        
        self.decoder_refer = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        
        self.decoder_glass = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )

        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

        
        self.adain = AdaIN()
        self.masking = nn.Parameter(torch.zeros(1,1,512,512))
        # self.noise_trans = nn.Parameter(torch.randn(1,ndf,64, 64).cuda())
        # self.generating = nn.Sequential(
        #     nn.ConvTranspose2d(ndf, ndf*4, 3,2,0, bias=False),  #128
        #     nn.ReLU(), 
        #     nn.BatchNorm2d(ndf*4),
        #     nn.Conv2d(ndf*4, ndf*4, 3,1,0, bias=False),
        #     nn.ReLU(), 

        #     nn.ConvTranspose2d(ndf*4, ndf*2, 3,2,0, bias=False),  #256
        #     nn.ReLU(), 
        #     # nn.BatchNorm2d(ndf*2),
        #     nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False),
        #     nn.ReLU(), 
    
        #     nn.ConvTranspose2d(ndf*2, ndf*2, 3,2,0, bias=False),  #512
        #     nn.ReLU(),
        #     # nn.BatchNorm2d(ndf*2),
        #     nn.Conv2d(ndf*2, ndf, 3,1,0, bias=False),
        #     nn.ReLU(),   
        # )

        self.trans_mean_input = nn.Parameter(torch.zeros(1,ndf, 1,1))
        self.trans_mean_layer = nn.Sequential(
            nn.Conv2d(ndf, ndf*4, 1,1,0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf*4, ndf, 1,1,0, bias=False),
            
        )

        self.noise_trans = nn.Parameter(torch.randn(1,ndf, 128, 128).cuda())    
        self.deconv1 = nn.Sequential(
            nn.Conv2d(ndf, ndf*4, 3,1,0, bias=False),  #32
            nn.ReLU(),
            nn.BatchNorm2d(ndf*4),
            nn.Conv2d(ndf*4, ndf*4, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*2, 3,1,0, bias=False), #64
            nn.ReLU(),
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False),  #128
            nn.ReLU(),
            
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf*2, ndf, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )

    def generating(self, noise):
        y = self.deconv1(noise)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv2(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv3(y)
        y = F.interpolate(y, self.output_size, mode='bilinear', align_corners=False)
        
        return y 

    def return_mask(self):
        return self.masking * 1

    def return_trans_score(self):
        return 1*self.trans_mean_layer(self.trans_mean_input)

    def predict_reflection(self, source, target):
        _, _, h, w = source.shape

        refer_feat = self.encoder(target)
        glass_feat = self.encoder(source)
        
        refle_feat = torch.sigmoid(self.masking) * glass_feat
        refle_feat_adain = self.adain(refle_feat, refer_feat)
        refle_feat_adain = F.relu(refle_feat_adain)
        refle = self.decoder(refle_feat_adain)
        
        return refle 

    def auto_encoder(self, glass):
        glass_feat = self.encoder(glass)
        glass = self.decoder(glass_feat)
        return glass

    def synthesize_glass(self, refer, glass):
        _, _, h, w = refer.shape
        with torch.no_grad():
            glass_feat = self.encoder(glass)
            refer_feat = self.encoder(refer)
        
        trans_feat = self.generating(self.noise_trans)
        trans_feat = F.interpolate(trans_feat, (512,512), mode='bilinear', align_corners=False)
        
        recon_feat = trans_feat + refer_feat
        recon_std = torch.std(recon_feat, dim=[2,3], keepdim=True)
        glass_std = torch.std(glass_feat, dim=[2,3], keepdim=True)
        trans_mean = torch.mean(trans_feat, dim=[2,3], keepdim=True)
        glass_mean = torch.mean(glass_feat, dim=[2,3], keepdim=True)
        refer_mean = torch.mean(refer_feat, dim=[2,3], keepdim=True)

        trans_mask = torch.sigmoid(self.trans_mean_layer(self.trans_mean_input))
        refer_mask = torch.sigmoid(-self.trans_mean_layer(self.trans_mean_input))
        trans_feat_adain = (trans_feat - trans_mean) / recon_std * glass_std + 1*glass_mean*trans_mask
        refer_feat_adain = (refer_feat - refer_mean) / recon_std * glass_std + 1*glass_mean*refer_mask
        recon_feat_adain = trans_feat_adain + refer_feat_adain
        recon = self.decoder(recon_feat_adain)

        trans = self.decoder(trans_feat_adain)
        refle = self.decoder(refer_feat_adain)
        return recon,  trans, refle

    # def predict_transmission(self):
    #     trans_feat = self.generating(self.noise_trans)
    #     trans = self.decoder(trans_feat)
    #     return trans 


    def forward(self, source, target):
        _, _, h, w = source.shape

        refer_feat = self.encoder(target)
        glass_feat = self.encoder(source)
        glass_feat = F.interpolate(glass_feat, (512,512), mode='bilinear', align_corners=False)
        
        refle_feat = torch.sigmoid(self.masking) * glass_feat
        refle_feat_adain = self.adain(refle_feat, refer_feat)
        refle_feat_adain = F.relu(refle_feat_adain)
        refle = self.decoder_glass(refle_feat_adain)
        
        refer_feat_aligned = self.encoder(refle)
        trans_feat = self.generating(self.noise_trans)
        trans_feat = F.interpolate(trans_feat, (512,512), mode='bilinear', align_corners=False)
        
        glass_recon_feat = trans_feat + refer_feat_aligned
        glass_recon_feat_adain = self.adain(glass_recon_feat, glass_feat)
        trans_mean = torch.mean(glass_feat, dim=[2,3], keepdim=True) - torch.mean(refle_feat, dim=[2,3], keepdim=True)
        trans_std = torch.std(glass_feat, dim=[2,3], keepdim=True) + torch.std(refle_feat, dim=[2,3], keepdim=True) \
                    - torch.std(glass_feat, dim=[2,3], keepdim=True)* torch.std(refle_feat, dim=[2,3], keepdim=True)
        trans_feat = self.adain.forward2(trans_feat, trans_mean, trans_std.clamp(0.1))

        glass = self.decoder_glass(glass_recon_feat_adain)
        trans = self.decoder_glass(trans_feat)
        
        return trans, refle, glass


    def auto_encoder2(self, glass, refer):
        
        glass_feat = self.encoder(glass)
        glass_feat = F.interpolate(glass_feat, (512,512), mode='bilinear', align_corners=False)
        glass = self.decoder_glass(glass_feat)
        
        refer_feat = self.encoder(refer)
        refer_feat = F.interpolate(refer_feat, (512,512), mode='bilinear', align_corners=False)
        refer = self.decoder_refer(refer_feat)

        glass2 = self.decoder_refer(glass_feat)
        return glass, refer, glass2


###------------------------------------------------------------
class GeneratorSub(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64, output_size=(512,512)):
        """Construct a 1x1 PatchGAN discriminator"""
        super(GeneratorSub, self).__init__()
        self.output_size = output_size
        self.ndf = ndf
        self.deconv1 = nn.Sequential(
            nn.Conv2d(ndf, ndf*4, 3,1,0, bias=False),  #32
            nn.ReLU(),
            nn.BatchNorm2d(ndf*4),
            nn.Conv2d(ndf*4, ndf*4, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*2, 3,1,0, bias=False), #64
            nn.ReLU(),
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False),  #128
            nn.ReLU(),
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf*2, ndf, 3,1,0, bias=False),  #32
            # nn.Sigmoid(), 
        )
    def forward(self, noise):
        y = self.deconv1(noise)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv2(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv3(y)
        y = F.interpolate(y, self.output_size, mode='bilinear', align_corners=False)
        return y 


class Generator4(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Generator4, self).__init__()
        self.ndf = ndf
        self.output_size=(512,512)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        
        self.encoder_trans = nn.Sequential(
            nn.Conv2d(6, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder_trans = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        

        self.adain = AdaIN()
        
        self.noise_trans = nn.Parameter(torch.randn(1,ndf, 128, 128).cuda())    
        self.noise_refle = nn.Parameter(torch.randn(1,ndf, 128, 128).cuda())    
        
        self.trans_generator = GeneratorSub(self.ndf, self.output_size)
        self.refle_generator = GeneratorSub(self.ndf, self.output_size)
        

    def auto_encoder(self, input):
        input_feat = self.encoder(input)
        output = self.decoder(input_feat)
        return output


    def forward(self, glass, refer):
        _, _, h, w = refer.shape
        with torch.no_grad():
            refer_feat = self.encoder(refer)
            glass_feat = self.encoder(glass)
        

        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        refle_feat = glass_feat * torch.sigmoid(refle_mask)

        # trans_mask = self.trans_generator(self.noise_trans)
        # trans_mask = F.interpolate(trans_mask, (512,512), mode='bilinear', align_corners=False)
        trans_feat = glass_feat * torch.sigmoid(-refle_mask)

        recon = self.decoder(glass_feat)
        trans = self.decoder(trans_feat)
        refle = self.decoder(refle_feat)
        refle_recov_feat = self.adain(refle_feat, refer_feat)
        refle_recov = self.decoder(refle_recov_feat)
        
        return recon, trans, refle, refle_recov

    def synthesis(self, trans, refle):
        trans_feat = self.encoder(trans)
        refle_feat = self.encoder(refle)
        trans_feat = F.interpolate(trans_feat, (512,512), mode='bilinear', align_corners=False)
        refle_feat = F.interpolate(refle_feat, (512,512), mode='bilinear', align_corners=False)
        glass = self.decoder(trans_feat + refle_feat)
        return glass

    
    def predict(self, glass):
        
        glass_feat = self.encoder(glass)
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        trans_feat = glass_feat * torch.sigmoid(-refle_mask)
        trans = self.decoder(trans_feat)        
        refle_feat = glass_feat * torch.sigmoid(refle_mask)
        refle = self.decoder(refle_feat)        
        return trans, refle

    
    def predict_trans(self, glass, refer):
        feat = self.encoder(torch.cat((glass,refer), 1))
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        trans_feat = feat * torch.sigmoid(-refle_mask)
        trans = self.decoder(trans_feat)        
        return trans
    
    # def forward(self, glass, refer):
    #     _, _, h, w = refer.shape
    #     with torch.no_grad():
    #         refer_feat = self.encoder(refer)
    #         glass_feat = self.encoder(glass)
        
    #     trans_feat = self.trans_generator(self.noise_trans)
    #     trans_feat = F.interpolate(trans_feat, (512,512), mode='bilinear', align_corners=False)
        
    #     refle_feat = self.refle_generator(self.noise_refle)
    #     refle_feat = F.interpolate(refle_feat, (512,512), mode='bilinear', align_corners=False)
    #     # refle_recov_feat = self.adain(refle_feat, refer_feat)
        
    #     trans_mean = torch.mean(trans_feat, dim=[2,3], keepdim=True)
    #     refle_mean = torch.mean(refle_feat, dim=[2,3], keepdim=True)
    #     glass_mean_t = self.trans_mean_layer(glass_feat)
    #     glass_mean_r = self.refle_mean_layer(glass_feat)
    #     glass_std = torch.std(glass_feat, dim=[2,3], keepdim=True)
    #     feat_sum_std = torch.std(trans_feat + refle_feat, dim=[2,3], keepdim=True)

    #     trans_enhance_feat = (trans_feat - trans_mean) / feat_sum_std * glass_std + glass_mean_t
    #     refle_enhance_feat = (refle_feat - refle_mean) / feat_sum_std * glass_std + glass_mean_r
    #     glass_recon_feat = trans_enhance_feat + refle_enhance_feat

    #     recon = self.decoder(glass_recon_feat)
    #     trans = self.decoder(trans_enhance_feat)
    #     refle = self.decoder(refle_enhance_feat)
    #     refle_recov_feat = self.adain(refle_enhance_feat, refer_feat)
    #     refle_recov = self.decoder(refle_recov_feat)
        
    #     return recon, trans, refle, refle_recov





class Generator5(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Generator5, self).__init__()
        self.ndf = ndf
        self.output_size=(512,512)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        
        self.encoder_trans = nn.Sequential(
            nn.Conv2d(6, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder_trans = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(),
        )
        

        self.adain = AdaIN()
        
        self.noise_trans = nn.Parameter(torch.randn(1,ndf, 128, 128).cuda())    
        self.noise_refle = nn.Parameter(torch.randn(1,ndf, 128, 128).cuda())    
        
        self.trans_generator = GeneratorSub(self.ndf, self.output_size)
        self.refle_generator = GeneratorSub(self.ndf, self.output_size)
        

    def auto_encoder(self, input):
        input_feat = self.encoder(input)
        output = self.decoder(input_feat)
        return output
    

    # def auto_encoder2(self, input):
    #     input = torch.cat((input, torch.zeros_like(input)), 1)
    #     input_feat = self.encoder_trans(input)
    #     output = self.decoder_trans(input_feat)
    #     return output
    
    # def auto_encoder3(self, input):
    #     input = torch.cat((input, input), 1)
    #     input_feat = self.encoder_trans(input)
    #     output = self.decoder_trans(input_feat)
    #     return output

    def forward_refle(self, glass, refer):
        _, _, h, w = refer.shape
        with torch.no_grad():
            refer_feat = self.encoder(refer)
            glass_feat = self.encoder(glass)
        
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        refle_feat = glass_feat * torch.sigmoid(refle_mask)

        refle = self.decoder(refle_feat)
        refle_recov_feat = F.relu(self.adain(refle_feat, refer_feat))
        refle_recov = self.decoder(refle_recov_feat)
        return refle, refle_recov

    
    
    # def removal(self, glass):
    #     feat = self.encoder_trans(torch.cat((glass, refer), 1))
    #     trans_mask = self.trans_generator(self.noise_trans)
    #     trans_mask = F.interpolate(trans_mask, (512,512), mode='bilinear', align_corners=False)
    #     trans_feat = feat * torch.sigmoid(trans_mask)
    #     trans = self.decoder_trans(trans_feat)        
    #     return glass_recon, trans


    # def forward_trans(self, glass, refer):

    #     feat = self.encoder_trans(torch.cat((glass, refer), 1))
    #     refle_mask = self.refle_generator(self.noise_refle)
    #     refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
    #     trans_feat = feat * torch.sigmoid(-refle_mask)
    #     trans = self.decoder(trans_feat)        
        
    #     return trans

    
    def forward_refle(self, glass, refer):
        _, _, h, w = refer.shape
        with torch.no_grad():
            refer_feat = self.encoder(refer)
            glass_feat = self.encoder(glass)
        
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        refle_feat = glass_feat * torch.sigmoid(refle_mask)
        refle = self.decoder(refle_feat)

        refle_recov_feat = F.relu(self.adain(refle_feat, refer_feat))
        refle_recov = self.decoder(refle_recov_feat)

        return refle, refle_recov

    def forward(self, glass, refer):
        _, _, h, w = refer.shape
        with torch.no_grad():
            refer_feat = self.encoder(refer)
            glass_feat = self.encoder(glass)
        
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        refle_feat = glass_feat * torch.sigmoid(refle_mask)
        refle = self.decoder(refle_feat)

        trans_mask = self.trans_generator(self.noise_trans)
        trans_mask = F.interpolate(trans_mask, (512,512), mode='bilinear', align_corners=False)
        trans_feat = glass_feat * torch.sigmoid(trans_mask)
        # trans_feat = trans_feat - torch.mean(trans_feat, dim=[1,2,3], keepdim=True) + torch.mean(glass_feat, dim=[1,2,3], keepdim=True))
        trans = self.decoder(trans_feat)
        
        
        refle_recov_feat = F.relu(self.adain(refle_feat, refer_feat))
        refle_recov = self.decoder(refle_recov_feat)

        return trans, refle, refle_recov

    



    def synthesis(self, trans, refer,scale):
        trans_feat = self.encoder(trans)
        refle_feat = self.encoder(refer) * scale
        trans_feat = F.interpolate(trans_feat, (512,512), mode='bilinear', align_corners=False)
        refle_feat = F.interpolate(refle_feat, (512,512), mode='bilinear', align_corners=False)
        glass = self.decoder(trans_feat + refle_feat)
        return glass

    


    def synthesis2(self, trans, refer, glass):
        trans_feat = self.encoder(trans)
        refle_feat = self.encoder(refer) 
        glass_feat = self.encoder(glass)
        syn_feat = F.relu(self.adain(trans_feat + refle_feat, glass_feat))

        syn = self.decoder(syn_feat)
        return syn

    def synthesis3(self, trans, refer):
        trans_feat = self.encoder(trans)
        refle_feat = self.encoder(refer) 
        syn = self.decoder(trans_feat + refle_feat)
        return syn

    
    def synthesis4(self, trans, refle, trans_refer, refle_refer):
        trans_feat = self.encoder(trans)
        refle_feat = self.encoder(refle) 
        trans_refer_feat = self.encoder(trans_refer) 
        refle_refer_feat = self.encoder(refle_refer) 
        syn_feat = F.relu(self.adain(trans_feat, trans_refer_feat)) + F.relu(self.adain(refle_feat, refle_refer_feat))
        syn = self.decoder(syn_feat)
        return syn

        
    
    def predict(self, glass):
        
        glass_feat = self.encoder(glass)
        refle_mask = self.refle_generator(self.noise_refle)
        refle_mask = F.interpolate(refle_mask, (512,512), mode='bilinear', align_corners=False)
        trans_feat = glass_feat * torch.sigmoid(-refle_mask)
        trans = self.decoder(trans_feat)        
        refle_feat = glass_feat * torch.sigmoid(refle_mask)
        refle = self.decoder(refle_feat)        
        return trans, refle


class FusionNet(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, num_images=10, image_size=(512,512)):
        """Construct a 1x1 PatchGAN discriminator"""
        super(FusionNet, self).__init__()
        self.scores = nn.Parameter(torch.zeros(1,num_images, image_size[0], image_size[1]))
        self.labels = torch.linspace(0,num_images-1, num_images).cuda()
        
    def forward(self, input):
        input = input.permute(1,0,2,3)
        prob = F.softmax(self.scores, dim=1)
        output = torch.sum(input * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output

    def return_prob(self):
        return F.softmax(self.scores, dim=1)
    
    def return_scores(self):
        return  1 * self.scores
    
    def return_softlabel(self):
        prob = F.softmax(self.scores, dim=1)
        softlabel = F.conv2d(prob, self.labels.view(1,-1,1,1))
        return softlabel

    def gather(self, input, prob):
        input = input.permute(1,0,2,3)
        output = torch.sum(input * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output

class FusionNet2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, num_images=10, rolling=5, image_size=(512,512)):
        """Construct a 1x1 PatchGAN discriminator"""
        super(FusionNet2, self).__init__()
        total_num_images = num_images*((2*rolling+1)**2)
        self.scores = nn.Parameter(torch.zeros(1, total_num_images, image_size[0], image_size[1]))
        self.rolling = rolling
        self.num_images = num_images

    def forward(self, input):
        nb, nc, h, w = input.shape
        input = input.permute(1,0,2,3)
        prob = F.softmax(self.scores, dim=1)
        prob_list = torch.split(prob, self.num_images, 1)
        output = torch.zeros(nc, 1, h, w).to(input.device)
        cnt = 0
        for i in range(-self.rolling, self.rolling+1):
            for j in range(-self.rolling, self.rolling+1):
                output += torch.sum(prob_list[cnt] * torch.roll(input, (i,j), (2,3)), dim=1, keepdim=True)
                cnt += 1
                
        output = output.permute(1,0,2,3)

        return output

    def return_prob(self):
        return F.softmax(self.scores, dim=1)
    
    def return_scores(self):
        return  1 * self.scores
    
    def return_softlabel(self, labels):
        prob = F.softmax(self.scores, dim=1)
        softlabel = F.conv2d(prob, labels.view(1,-1,1,1))
        return softlabel

    def gather(self, input, prob):
        input = input.permute(1,0,2,3)
        output = torch.sum(input * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output

class PatchExpander(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, input_channels, patch_size):
        """Construct a 1x1 PatchGAN discriminator"""
        super(PatchExpander, self).__init__()
        
        self.patch_size = patch_size 

        self.kernel = nn.Conv2d(input_channels, patch_size*patch_size*input_channels, patch_size, patch_size, bias=False)
        cnt = 0
        for c in range(input_channels):
            for i in range(patch_size):
                for j in range(patch_size):
                    self.kernel.weight.data[cnt, c, i, j] = 1
                    cnt += 1
        self.pad = nn.ReplicationPad2d(patch_size//2)
    def forward(self, input):
        input = self.pad(input)
        output = self.kernel(input)
        
        return output


class MaskGenerator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=32, output_size=(512,512)):
        """Construct a 1x1 PatchGAN discriminator"""
        super(MaskGenerator, self).__init__()
        self.ndf = ndf
        self.output_size = output_size
        self.deconv1 = nn.Sequential(
            nn.Conv2d(ndf, ndf*4, 3,1,0, bias=False),  #32
            nn.ReLU(), 
        )
        self.deconv2 = nn.Sequential(
            nn.BatchNorm2d(ndf*4),
            nn.Conv2d(ndf*4, ndf*4, 3,1,0, bias=False), #64
            nn.ReLU(), 
        )
        self.deconv3 = nn.Sequential(
            nn.BatchNorm2d(ndf*4),
            nn.Conv2d(ndf*4, ndf*2, 3,1,0, bias=False),  #128
            nn.ReLU(), 
        )
        self.deconv4 = nn.Sequential(
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf*2, ndf*2, 3,1,0, bias=False), #256
            nn.ReLU(), 
            nn.Conv2d(ndf*2, 1, 3,1,0, bias=False),  #512
            nn.Sigmoid(),
        )
        
        self.noise = torch.randn(1,ndf, 64,64).cuda()

    def forward(self):
        y = self.deconv1(self.noise)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv2(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv3(y)
        y = F.interpolate(y, (y.shape[2]*2, y.shape[3]*2), mode='bilinear', align_corners=False)
        y = self.deconv4(y)
        y = F.interpolate(y, self.output_size, mode='bilinear', align_corners=False)
        return y

class Generator3(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Generator3, self).__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
        )
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)
        
        self.adain = AdaIN()
        # self.masking_refle = nn.Parameter(torch.zeros(1,1,512,512))
        # self.masking_trans = nn.Parameter(torch.zeros(1,1,512,512))
        # self.noise_trans = nn.Parameter(torch.randn(1,ndf,64, 64).cuda())
        # self.noise_refle = nn.Parameter(torch.randn(1,ndf,64, 64).cuda())
        
        self.generate_trans_mask = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        
        self.generate_refle_mask = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf*4, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        
        self.gsl10 = GSL10.apply

    def return_mask(self):
        return self.masking_refle * 1


    def forward(self, glass, refer):
        _, _, h, w = glass.shape
        
        glass_feat = self.encoder(glass)
        refer_feat = self.encoder(refer)

        masking_trans = self.generate_trans_mask(glass)
        masking_refle = self.generate_refle_mask(glass)
        # masking_trans = F.interpolate(masking_trans, (512, 512), mode='bilinear', align_corners=False)
        # masking_refle = F.interpolate(masking_refle, (512, 512), mode='bilinear', align_corners=False)
        trans_feat = masking_trans * glass_feat
        refle_feat = masking_refle * glass_feat
        
        glass_feat = (masking_trans+masking_refle).clamp(max=1) * glass_feat
        refle_feat_adain = self.adain(refle_feat, refer_feat)
        
        trans = self.decoder(trans_feat)
        recon = self.decoder(glass_feat)
        refle = self.decoder(refle_feat_adain)
        return recon, trans, refle



class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, reduction='mean'):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if reduction == 'mean':
            b = -1.0 * b.mean()
        elif reduction == 'sum':
            b = -1.0 * b.sum()
        else:
            b = -1.0 * b
        return b

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, x, y):
        eps = 1e-5	
        mean_x = torch.mean(x, dim=[2,3])
        mean_y = torch.mean(y, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])
        std_y = torch.std(y, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x)/ std_x * std_y + mean_y

        return out

    def forward2(self, x, mean_y, std_y):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        if len(mean_y.shape) != 4:
            mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        if len(std_y.shape) != 4:
            std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x)/ std_x * std_y + mean_y

        return out



class AdaIN2(nn.Module):
    def __init__(self):
        super(AdaIN2, self).__init__()

    def forward(self, x, mean_y, std_y):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2,3])
        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)

        std_x = torch.std(x, dim=[2,3])
        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        std_y = std_y + eps

        out = (x - mean_x)/ std_x * std_y + mean_y
        return out

class PixAdaIN(nn.Module):
	def __init__(self):
		super(PixAdaIN, self).__init__()

	def forward(self, x, y):
		eps = 1e-5	
		mean_x = torch.mean(x, dim=1, keepdim=True)
		mean_y = torch.mean(y, dim=1, keepdim=True)

		std_x = torch.std(x, dim=1, keepdim=True)
		std_y = torch.std(y, dim=1, keepdim=True)

		std_x = std_x + eps
		std_y = std_y + eps

		out = (x - mean_x)/ std_x * std_y + mean_y

		return out

class Refer2Refle(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Refer2Refle, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(10, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(self.ndf*4, 9, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        # self.auto_encoder[0].weight.div(self.ndf)
        # self.auto_encoder[2].weight.div(self.ndf)
        # self.auto_encoder[4].weight.div(self.ndf)
        
        self.blur_layer = Blur(5)
        bias_h = torch.linspace(0,2,512).view(1,1,-1,1)
        bias_c = torch.linspace(0,2,9).view(1,-1,1,1)
        bias = torch.exp(-5*((bias_h - bias_c) ** 2.))
        bias = bias / torch.max(bias, dim=1, keepdim=True)[0]
        self.bias = bias.repeat(1,1,1,512).cuda()

    def forward(self, source, input, train=True):
        # if train:
        #     if random.random() > 0.5:
        #         input_blur = self.blur_layer(input)
        #     else:
        #         input_blur = input
        # else:
        #     input_blur = input
        input_blur = input
        input2 = input_blur.permute(1,0,2,3)
        if train:
            input2 = input2 + torch.randn(input2.shape).cuda() / 10.
        # source2 = source.repeat(len(input2), 1,1,1)
        Y = torch.linspace(-1, 1, input2.shape[2]).view(1,1,-1,1).repeat(input2.shape[0],1,1,input2.shape[3]).cuda()
        X = torch.linspace(-1, 1, input2.shape[3]).view(1,1,1,-1).repeat(input2.shape[0],1,input2.shape[2],1).cuda()
        XY = torch.cat((input2, X,Y), 1)
        

        # if train:
        #     XY = XY + torch.randn(XY.shape).cuda() / 10.
        
        weight = self.auto_encoder(XY)
        weight = torch.mean(weight, dim=0, keepdim=True)
        weight = weight + self.bias
        # print(weight)
        # print(weight[0,:,:10,:10])
        prob = F.softmax(1*weight, dim=1)
        output = torch.sum(input.permute(1,0,2,3) * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output, prob, weight
    

class Refer2Refle2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Refer2Refle2, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(11, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(self.ndf*4, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        
        bias_h = torch.linspace(0,2,512).view(1,1,-1,1)
        bias_c = torch.linspace(0,2,10).view(1,-1,1,1)
        bias = torch.exp(-5*((bias_h - bias_c) ** 2.))
        bias = bias / torch.max(bias, dim=1, keepdim=True)[0]
        self.bias = bias.repeat(1,1,1,512).cuda()
        
    def forward(self, source, input, train=True):
        
        input2 = torch.cat((source,input), 0).permute(1,0,2,3)
        Y = torch.linspace(-1, 1, input2.shape[2]).view(1,1,-1,1).repeat(input2.shape[0],1,1,input2.shape[3]).cuda()
        X = torch.linspace(-1, 1, input2.shape[3]).view(1,1,1,-1).repeat(input2.shape[0],1,input2.shape[2],1).cuda()
        XY = torch.cat((input2, X,Y), 1)
        
        if train:
            input2 = input2 + torch.randn(input2.shape).cuda() / 10.
        
        weight = self.auto_encoder(input2)
        weight = torch.mean(weight, dim=0, keepdim=True)
        weight = weight + self.bias
        
        prob = F.softmax(1*weight, dim=1)
        output = torch.sum(input.permute(1,0,2,3) * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output, prob, weight


class Refer2Refle4(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Refer2Refle4, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(12, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(self.ndf*4, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        
    def forward(self, refer, train=True):
        
        refer2 = refer.permute(1,0,2,3)
        if train:
            refer2 = refer2 + torch.randn(refer2.shape).cuda() / 20.
        # source2 = source.repeat(len(refer2), 1,1,1)
        Y = torch.linspace(-1, 1, refer2.shape[2]).view(1,1,-1,1).repeat(refer2.shape[0],1,1,refer2.shape[3]).cuda()
        X = torch.linspace(-1, 1, refer2.shape[3]).view(1,1,1,-1).repeat(refer2.shape[0],1,refer2.shape[2],1).cuda()
        input = torch.cat((refer2, X,Y), 1)
        
        weight = self.auto_encoder(input)
        weight = torch.mean(weight, dim=0, keepdim=True)
        
        prob = F.softmax(1*weight, dim=1)
        output = torch.sum(refer.permute(1,0,2,3) * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output, prob, weight



class Refer2Refle3(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(Refer2Refle3, self).__init__()
        self.ndf = ndf
        self.auto_encoder = nn.Sequential(
            nn.Conv2d(3, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            # nn.Conv2d(self.ndf*4, self.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(self.ndf*4, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )
        
        bias_h = torch.linspace(0,2,512).view(1,1,-1,1)
        bias_c = torch.linspace(0,2,10).view(1,-1,1,1)
        bias = torch.exp(-5*((bias_h - bias_c) ** 2.))
        bias = bias / torch.max(bias, dim=1, keepdim=True)[0]
        self.bias = bias.repeat(1,1,1,512).cuda()
        
        # self.adain = AdaIN()
    def forward(self, source, input, train=True):
        
        input2 = torch.cat((source,input), 0).permute(1,0,2,3)

        if train:
            input2 = input2 + torch.randn(input2.shape).cuda() / 10.
        
        weight_refer = self.auto_encoder(input)
        weight_source = self.auto_encoder(source)
        sim = torch.sum(weight_refer * weight_source, dim=1, keepdim=True)
        sim = sim.permute(1,0,2,3)
        
        prob = F.softmax(1*sim, dim=1)
        output = torch.sum(input.permute(1,0,2,3) * prob, dim=1, keepdim=True)
        output = output.permute(1,0,2,3)
        return output, prob, sim
        
class ColorRecon(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=16):
        """Construct a 1x1 PatchGAN discriminator"""
        super(ColorRecon, self).__init__()
        self.ndf = ndf
        self.encode = nn.Sequential(
            nn.Conv2d(6, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
        )
        
        self.auto_decoder_layer1 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.auto_decoder_layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf*4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, self.ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        
        self.auto_decoder_layer3 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.ndf*4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
    
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        
    def gradient(self, x):
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, padding=(0,1,1)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, padding=(0,1,1)).squeeze(1)
        grad = torch.cat((dx,dy), 1)
        return grad

    def decode(self, feat, output_shape):
        y1 = self.auto_decoder_layer1(feat)
        y1 = F.interpolate(y1, (y1.shape[2]*2,y1.shape[3]*2), mode='bilinear', align_corners=False)
        y2 = self.auto_decoder_layer2(y1)
        y2 = F.interpolate(y2, (y2.shape[2]*2,y2.shape[3]*2), mode='bilinear', align_corners=False)
        y3 = self.auto_decoder_layer3(y2)
        output = F.interpolate(y3, output_shape, mode='bilinear', align_corners=False)
        return output

    def forward(self, input):
        feat = self.encode(input)
        output = self.decode(feat, input.shape[-2:])
        return output
    


class KernelDiction(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(KernelDiction, self).__init__()
        self.ndf = ndf
        self.kernel = nn.Parameter(torch.randn(32, ndf*3, 1, 1)/ndf)
        
        mapping = torch.zeros(ndf, 1, 8, 8)
        for k in range(ndf):
            for c in range(1):
                for h in range(8):
                    for w in range(8):
                        mapping[k,c,h,w] = 1
        self.mapping = mapping.detach().clone().cuda()

        self.batchnorm = nn.BatchNorm2d(ndf*3, affine=False)
    def forward(self, input):
        """Standard forward."""
        
        r,g,b = torch.split(input, 1,1)
        rf = F.conv2d(r, self.mapping, bias=None)
        gf = F.conv2d(g, self.mapping, bias=None)
        bf = F.conv2d(b, self.mapping, bias=None)        
        feat = torch.cat((rf, gf, bf), 1)
        feat = feat / torch.norm(feat, dim=1, keepdim=True).clamp(min=1e-9)
        diverse_kernel = self.batchnorm(self.kernel)
        kernel =  diverse_kernel / torch.norm(diverse_kernel, dim=1, keepdim=True).clamp(min=1e-9)
        output = F.conv2d(feat, kernel, bias=None)
        return output



class KernelDiction(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(KernelDiction, self).__init__()
        self.ndf = ndf
        self.kernel = nn.Parameter(torch.randn(32, ndf*3, 1, 1)/ndf)
        
        mapping = torch.zeros(ndf, 1, 8, 8)
        for k in range(ndf):
            for c in range(1):
                for h in range(8):
                    for w in range(8):
                        mapping[k,c,h,w] = 1
        self.mapping = mapping.detach().clone().cuda()

        self.batchnorm = nn.BatchNorm2d(ndf*3, affine=False)
    def forward(self, input):
        """Standard forward."""
        
        r,g,b = torch.split(input, 1,1)
        rf = F.conv2d(r, self.mapping, bias=None)
        gf = F.conv2d(g, self.mapping, bias=None)
        bf = F.conv2d(b, self.mapping, bias=None)        
        feat = torch.cat((rf, gf, bf), 1)
        feat = feat / torch.norm(feat, dim=1, keepdim=True).clamp(min=1e-9)
        diverse_kernel = self.batchnorm(self.kernel)
        kernel =  diverse_kernel / torch.norm(diverse_kernel, dim=1, keepdim=True).clamp(min=1e-9)
        output = F.conv2d(feat, kernel, bias=None)
        return output


class KernelDiction2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=256):
        """Construct a 1x1 PatchGAN discriminator"""
        super(KernelDiction2, self).__init__()
        self.ndf = ndf
        self.encode = nn.Sequential(
            nn.Conv2d(6, ndf//4, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf//4, ndf//2, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf//2, ndf, 1, 1, 0, bias=False),
        )
        self.kernel = nn.Parameter(torch.randn(64, ndf, 1,1))

        self.batchnorm = nn.BatchNorm2d(ndf, affine=True)
        self.color_grad_layer = GradientLayer().cuda()
    def forward(self, input):
        """Standard forward."""
        input_grad = self.color_grad_layer(input, 'none')
        feat = self.encode(torch.cat((input, input_grad), 1))
        feat = feat / torch.norm(feat, dim=1, keepdim=True).clamp(min=1e-9)
        diverse_kernel = self.batchnorm(self.kernel)
        kernel =  diverse_kernel / torch.norm(diverse_kernel, dim=1, keepdim=True).clamp(min=1e-9)
        output = F.conv2d(feat, kernel, bias=None)
        return output


class ExclusivePatch(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, ndf=64):
        """Construct a 1x1 PatchGAN discriminator"""
        super(ExclusivePatch, self).__init__()
        self.ndf = ndf
        
        self.matching = nn.Sequential( 
            nn.Conv2d(6, ndf, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf, ndf, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, grad1, grad2, train=True):
        """Standard forward."""    
        # grad1 = torch.abs(grad1)
        # grad2 = torch.abs(grad2)
        if not train:
            fake_score = self.matching(torch.cat((grad1, grad2), 1))
            return fake_score
        else:
            ZERO = torch.zeros_like(grad2)
            real_score1 = self.matching(torch.cat((grad1, ZERO), 1))
            real_score2 = self.matching(torch.cat((ZERO, grad2), 1))
            fake_score = self.matching(torch.cat((grad1, grad2), 1))

            return real_score1, real_score2, fake_score



class ImageDiction(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, img):
        """Construct a 1x1 PatchGAN discriminator"""
        super(ImageDiction, self).__init__()
        # self.image = img.detach().clone()
        # self.ndf = ndf
        self.image = nn.Parameter(img)
        # self.auto_encoder = nn.Sequential(
        #     nn.Conv2d(4, self.ndf, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.ndf, self.ndf*4, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(self.ndf*4, 3,   kernel_size=1, stride=1, padding=0, bias=False),
        # )
        # self.auto_encoder[0].weight.data.div(self.ndf*4)
        # self.auto_encoder[2].weight.data.div(self.ndf*4)
        # self.auto_encoder[4].weight.data.div(self.ndf*4)
    def forward(self, input=None):
        """Standard forward."""
        # output = self.image + self.auto_encoder(torch.cat((self.image, self.noise), 1))
        output = self.image * 1
        return output



class RGBshifter(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self):
        """Construct a 1x1 PatchGAN discriminator"""
        super(RGBshifter, self).__init__()
        
        self.mean = nn.Parameter(torch.zeros(1,1,1,1))
        
    def forward(self, img):
        """Standard forward."""
        img = img + self.mean
        return img



class PatchDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator, self).__init__()
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator2(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator2, self).__init__()
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(4,2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(4,2),
            nn.Conv2d(ndf * 2, 3, kernel_size=4, stride=2, padding=0, bias=False)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator3(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator3, self).__init__()
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),    
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator4(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator4, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),    
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),    
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),    
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=2, padding=0, bias=False),    
        )
        
    def forward(self, input):
        """Standard forward."""
        out1 = self.net1(input)
        out2 = self.net2(input)
        out3 = self.net3(input)
        out4 = self.net4(input)

        h, w = out1.shape[-2:]

        mask1 = torch.zeros_like(out1)
        mask1[:,:,:, :w//2] = 1

        mask2 = torch.zeros_like(out2)
        mask2[:,:,:h//2,:] = 1

        mask3 = torch.zeros_like(out3)
        mask3[:,:,:, w//2:] = 1

        mask4 = torch.zeros_like(out4)
        mask4[:,:,h//2:,:] = 1

        out = mask1.detach() * out1 + mask2.detach() * out2 \
                + mask3.detach() * out3 + mask4.detach() * out4

        return  out


class PatchDiscriminator5(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator5, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
        )
        
        self.decode = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False),    
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),    
            nn.ReLU(),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=False),    
        )
        self.pix_adain = PixAdaIN()

    def forward(self, source, target):
        """Standard forward."""
        source_feat = self.encode(source)
        target_feat = self.encode(target)
        trans_feat = self.pix_adain(source_feat, target_feat)
        out = self.decode(trans_feat)
        return  out



class PatchDiscriminator6(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator6, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc*2, ndf, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=0, bias=False),    
            nn.ReLU(),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0, bias=False),    
            nn.Sigmoid(),
        )
        self.gradient_layer = GradientLayer()
    def forward(self, input):
        """Standard forward."""
        input_grad = self.gradient_layer(input, 'none')
        out = self.net(torch.cat((input, input_grad), 1))
        return  out



class PatchDiscriminator7(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator7, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc*2, ndf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=0, bias=False),    
            nn.Sigmoid(),
        )
        self.gradient_layer = GradientLayer()
    def forward(self, input):
        """Standard forward."""
        input_grad = self.gradient_layer(input, 'none')
        out = self.net(torch.cat((input, input_grad), 1))
        return  out



class PatchDiscriminator8(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(PatchDiscriminator8, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc*2, ndf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2,2),
            nn.BatchNorm2d(ndf),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2,2),
            nn.BatchNorm2d(ndf*2),
            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=0, bias=False),    
            # nn.Sigmoid(),
        )
        self.gradient_layer = GradientLayer()
    def forward(self, input):
        """Standard forward."""
        input_grad = self.gradient_layer(input, 'none')
        out = self.net(torch.cat((input, input_grad), 1))
        return  out




class ImageDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
        """
        super(ImageDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc*2, ndf, kernel_size=4, stride=2, padding=0, bias=False), #256
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False), #128
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False),  #64  
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False),    #32
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False),    #16
            nn.ReLU(),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False),    #8
            nn.ReLU(),
            nn.Conv2d(ndf, 1, kernel_size=4, stride=2, padding=0, bias=False),    #4
            nn.Sigmoid(),
        )
        self.gradient_layer = GradientLayer()
    def forward(self, input):
        """Standard forward."""
        input_grad = self.gradient_layer(input, 'none')
        out = self.net(torch.cat((input, input_grad), 1))
        return  out


class GradientLoss2(nn.Module):
    def __init__(self):
        super(GradientLoss2, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x, reduction='mean',  p=1, dilation=1):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        dx = self.g_clip(dx)
        dy = self.g_clip(dy)
        if p== 1:
            grad = torch.abs(dx) + torch.abs(dy)
        else:
            grad = (dx**2 + dy**2).clamp(0.00001) ** 0.5

        if reduction == 'mean':
            return grad.mean()
        elif reduction =='sum':
            return grad.sum()
        else:
            return grad



class GradientLayerNoLimit(nn.Module):
    def __init__(self):
        super(GradientLayerNoLimit, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x, reduction='mean',  p=1, dilation=1):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        # dx = self.g_clip(dx)
        # dy = self.g_clip(dy)
        if p== 1:
            grad = torch.abs(dx) + torch.abs(dy)
        else:
            grad = (dx**2 + dy**2).clamp(0.00001) ** 0.5

        if reduction == 'mean':
            return grad.mean()
        elif reduction =='sum':
            return grad.sum()
        else:
            return grad



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

        self.mask5 = nn.Parameter(torch.zeros(1, 512, 32,32))
        self.mask4 = nn.Parameter(torch.zeros(1, 512, 64,64))
        self.mask3 = nn.Parameter(torch.zeros(1, 256, 128,128))
        self.mask2 = nn.Parameter(torch.zeros(1, 128, 256,256))
        self.mask1 = nn.Parameter(torch.zeros(1, 64, 512,512))
        self.adain = AdaIN()

    def return_mask(self):
        mask1 = 1*self.mask1
        mask2 = 1*self.mask2
        mask3 = 1*self.mask3
        mask4 = 1*self.mask4
        mask5 = 1*self.mask5
        return [mask1, mask2, mask3, mask4, mask5]

    def forward(self, input, refer):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y1 = self.inc(refer)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)

        x1_t = (1-F.sigmoid(self.mask1)) * x1
        x2_t = (1-F.sigmoid(self.mask2)) * x2
        x3_t = (1-F.sigmoid(self.mask3)) * x3
        x4_t = (1-F.sigmoid(self.mask4)) * x4
        x5_t = (1-F.sigmoid(self.mask5)) * x5

        x1_r = F.sigmoid(self.mask1) * x1
        x2_r = F.sigmoid(self.mask2) * x2
        x3_r = F.sigmoid(self.mask3) * x3
        x4_r = F.sigmoid(self.mask4) * x4
        x5_r = F.sigmoid(self.mask5) * x5
        
        x1_r = self.adain(x1_r, y1)
        x2_r = self.adain(x2_r, y2)
        x3_r = self.adain(x3_r, y3)
        x4_r = self.adain(x4_r, y4)
        x5_r = self.adain(x5_r, y5)

        x_t = self.up1(x5_t, x4_t)
        x_t = self.up2(x_t, x3_t)
        x_t = self.up3(x_t, x2_t)
        x_t = self.up4(x_t, x1_t)
        trans = self.outc(x_t)

        x_r = self.up1(x5_r, x4_r)
        x_r = self.up2(x_r, x3_r)
        x_r = self.up3(x_r, x2_r)
        x_r = self.up4(x_r, x1_r)
        refle = self.outc(x_r)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        glass = self.outc(x)

        return trans, refle, glass

    
    def auto_encoder(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        glass = self.outc(x)

        return glass

class GradientLayer(nn.Module):
    def __init__(self):
        super(GradientLayer, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x, reduction='mean',  p=1, dilation=1):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        # dx = self.g_clip(dx)
        # dy = self.g_clip(dy)
        if p== 1:
            grad = torch.abs(dx) + torch.abs(dy)
        else:
            grad = (dx**2 + dy**2).clamp(0.00001) ** 0.5

        if reduction == 'mean':
            return grad.mean()
        elif reduction =='sum':
            return grad.sum()
        else:
            return grad


class GradientConcatLayer(nn.Module):
    def __init__(self):
        super(GradientConcatLayer, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float() / 5.0
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)


    def compute_gradient(self, x):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel).squeeze(1)
        
        return torch.cat((dx,dy), 1)

    def forward(self, x,  single_scale=False):
        if single_scale:
            return self.compute_gradient(x)    
        else:
            _, _, h, w = x.shape
            grad1 = self.compute_gradient(x)    
            grad2 = self.compute_gradient(F.interpolate(x, (h//2, w//2), mode='bilinear', align_corners=False))    
            grad4 = self.compute_gradient(F.interpolate(x, (h//4, w//4), mode='bilinear', align_corners=False))    
            return grad1, grad2, grad4 


class GradientConcatLayer2(nn.Module):
    def __init__(self):
        super(GradientConcatLayer2, self).__init__()
        
        dx_kernel = torch.tensor([[-1,1]]).float() / 1.0
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 1, 2)
        dy_kernel = dy_kernel.view(1, 1, 1, 2, 1)
        
        self.dx_layer = nn.Conv3d(1,1,(1,1,2), bias=False)
        self.dx_layer.weight.data = dx_kernel

        self.dy_layer = nn.Conv3d(1,1,(1,2,1), bias=False)
        self.dy_layer.weight.data = dy_kernel
        
        self.pad_x = nn.ReplicationPad2d((0,1,0,0))
        self.pad_y = nn.ReplicationPad2d((0,0,1,0))


    def compute_gradient(self, input):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad_x(input).unsqueeze(1)
        dx = self.dx_layer(x).squeeze(1)
        y = self.pad_y(input).unsqueeze(1)
        dy = self.dy_layer(y).squeeze(1)
        
        return torch.cat((dx,dy), 1)

    def forward(self, x,  single_scale=False):
        if single_scale:
            return self.compute_gradient(x)    
        else:
            _, _, h, w = x.shape
            grad1 = self.compute_gradient(x)    
            grad2 = self.compute_gradient(F.interpolate(x, (h//2, w//2), mode='bilinear', align_corners=False))    
            grad4 = self.compute_gradient(F.interpolate(x, (h//4, w//4), mode='bilinear', align_corners=False))    
            return grad1, grad2, grad4 

class GradientLayerColorWeight(nn.Module):
    def __init__(self):
        super(GradientLayerColorWeight, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False
        self.pad = nn.ReplicationPad2d(1)
        self.weight = torch.tensor([1.,0.5,0.2]).float().view(1,3,1,1).cuda()
    def forward(self, x, reduction='mean',  p=1, dilation=1):
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.pad(x)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dx = dx * self.weight
        dy = dy * self.weight

        # dx = self.g_clip(dx)
        # dy = self.g_clip(dy)
        if p== 1:
            grad = torch.abs(dx) + torch.abs(dy)
        else:
            grad = (dx**2 + dy**2).clamp(0.00001) ** 0.5

        if reduction == 'mean':
            return grad.mean()
        elif reduction =='sum':
            return grad.sum()
        else:
            return grad

class SencondDerivativeLoss(nn.Module):
    def __init__(self):
        super(SencondDerivativeLoss, self).__init__()
        
        dx_kernel = torch.tensor([[-1., 0., 1.], [-3, 0, 3], [-1, 0, 1]]).float()
        dx_kernel = dx_kernel / 5.0
        # dx_kernel /= dx_kernel.sum()
        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)
        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()
        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False

    def forward(self, x, reduction='mean',  p=1, dilation=1):
        x = torch.mean(x, dim=1, keepdim=True)
        dx = F.conv3d(x.unsqueeze(1), self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x.unsqueeze(1), self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        dxx = F.conv3d(dx.unsqueeze(1), self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dxy = F.conv3d(dx.unsqueeze(1), self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        dyx = F.conv3d(dy.unsqueeze(1), self.dx_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        dyy = F.conv3d(dy.unsqueeze(1), self.dy_kernel, padding=(0,0,0), dilation=(1,dilation,dilation)).squeeze(1)
        
        # dxx = self.g_clip(dxx)
        # dxy = self.g_clip(dxy)
        # dyx = self.g_clip(dyx)
        # dyy = self.g_clip(dyy)
        if p== 1:
            grad = torch.abs(dxx) + torch.abs(dxy) + torch.abs(dyx) + torch.abs(dyy)
        else:
            grad = (dxx**2 + dxy**2 + dyx**2 + dyy**2).clamp(0.00001) ** 0.5

        if reduction == 'mean':
            return grad.mean()
        elif reduction =='sum':
            return grad.sum()
        else:
            return grad



class GradientLoss3(nn.Module):
    def __init__(self):
        super(GradientLoss3, self).__init__()
        
        dx_kernel = torch.tensor([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]).float()
        dx_kernel /= 11.

        dy_kernel = dx_kernel.clone().t()
        dx_kernel = dx_kernel.view(1, 1, 1, 3, 3)
        dy_kernel = dy_kernel.view(1, 1, 1, 3, 3)

        self.dx_kernel = dx_kernel.detach().clone().cuda()
        self.dy_kernel = dy_kernel.detach().clone().cuda()

        self.g_clip = GradientClipper3.apply
        self.dx_kernel.requries_grad_ = False
        self.dy_kernel.requries_grad_ = False

    def forward(self, x, reduction='mean',  p=1, dilation=1):
        x = torch.mean(x, dim=1, keepdim=True)
        x = x.unsqueeze(1)
        dx = F.conv3d(x, self.dx_kernel, dilation=(1,dilation,dilation)).squeeze(1)
        dy = F.conv3d(x, self.dy_kernel, dilation=(1,dilation,dilation)).squeeze(1)
        return torch.cat((dx,dy), 1)

class DeepDictionary(nn.Module):
    def __init__(self, img, patch_size):
        super(DeepDictionary, self).__init__()

        self.encode = Encoder(3, 64)
        self.encode1 = Encoder(3, 64)
        self.encode2 = Encoder(3, 64)
        self.encode3 = Encoder(3, 64)

        self.r_decoder = Decoder(img, patch_size)
        self.t_decoder = Decoder(img, patch_size)

        self.gsl = GSL.apply
        self.mask = nn.Parameter(torch.randn(1,img.shape[1],512,512)/10)
        self.ref_mask = nn.Parameter(torch.randn(1,img.shape[1],512,512)/10)

    def referencing(self, img):
        mask_ = torch.tanh(F.interpolate(self.ref_mask, img.shape[-2:], mode='bilinear', align_corners=False))
        # mask_ = F.interpolate(self.ref_mask, img.shape[-2:], mode='bilinear', align_corners=False)
        img_ = self.r_decoder(mask_)
        return img_, mask_

    def separating(self, glass):
        mask_ = torch.tanh(F.interpolate(self.mask, glass.shape[-2:], mode='bilinear', align_corners=False))
        feat_t = mask_
        feat_r = -mask_
        trans_ = self.t_decoder(feat_t)
        refle_ = self.r_decoder(feat_r)
        
        return trans_, feat_t, refle_, feat_r



class StructureDictionary(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(StructureDictionary, self).__init__()
        self.ndf = ndf
        self.encode1 = Encoder(input_channels, ndf)
        self.encode2 = Encoder(input_channels, ndf)
        
    def part_forward(self, img):
        edge1 = self.encode1(img)
        edge2 = self.encode2(img)
        return edge1, edge2

    def forward(self, img, mask1, mask2):
        edge1, edge2 = self.part_forward(img)
        edge1 = F.softmax(edge1, dim=1)[:,[1],:,:]
        edge2 = F.softmax(edge2, dim=1)[:,[1],:,:]

        edge = mask1.detach() * edge1 + mask2.detach() * edge2 
        return edge

        

        