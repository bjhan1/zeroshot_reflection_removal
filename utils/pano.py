#%%
import cv2
import numpy as np
import os 
import scipy.io as sio
import random
import csv
import os.path
import sys
import matplotlib.pyplot as plt

#%%

def imshow(img, figsize=5):
    fig = plt.figure(figsize=(figsize, figsize))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def circIndex(index, limit):
    for i in range(len(index)):
        if(index[i] < 0):
            index[i] = limit + index[i]
    return index
        
def crop_img(img, point, patch_size=100):
    h0 = point[0]-int(patch_size/2)
    h = np.arange(h0,h0+patch_size)
    h = circIndex(h, img.shape[0])
    w0 = point[1]-int(patch_size/2)
    w = np.arange(w0,w0+patch_size)
    w = circIndex(w, img.shape[1])
    W, H = np.meshgrid(w, h)
    patch_img = img[H, W, :]
    marked = img
    marked[H,W,0] = marked[H,W,0]+0.5
    return patch_img, marked

def cvtXYZ2Sph(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)        
    theta = np.arccos(z/r)
    phi = np.arctan(np.divide(y, x, out=np.ones_like(y)*1000, where=x!=0))
#     phi[np.logical_and(x<0, y>0)] = np.pi+phi[np.logical_and(x<0, y>0)]
#     phi[np.logical_and(x<0, y<0)] = np.pi+phi[np.logical_and(x<0, y<0)]
    phi[x<0] = np.pi+phi[x<0]
    phi[np.logical_and(x>0, y<0)] = np.pi*2 + phi[np.logical_and(x>0, y<0)]
    return [phi, theta, r]

def cvtSph2Pix(phi, theta, H, W):
    phi = phi/(np.pi*2)*W
    theta = theta/(np.pi)*H
    return [phi, theta]

def cvtXYZ2Pix(x,y,z, H, W):
    [phi, theta, r]= cvtXYZ2Sph(x, y, z)
    [phi, theta] = cvtSph2Pix(phi, theta, img.shape[0], img.shape[1])
    phi = phi.astype(int)
    theta = theta.astype(int)
    return [phi, theta]

def cvtPix2Sph(H, W, height, width):
    # phi: horizontal axis and [-180, 180]
    phi = (W/width)*360
    # theta: vertical axis and [-90, 90]
    theta = (H/height)*180
    return [np.radians(phi), np.radians(theta)]

def cvtSph2XYZ(phi, theta, r):
    # pi should be 0~2pi
    # theta shoulbe 0~pi
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return [x,y,z]

def cvtPix2XYZ(H, W, height, width):
    [phi, theta] = cvtPix2Sph(H, W, height, width)
    r = np.ones_like(phi)*1
    [X, Y, Z] = cvtSph2XYZ(phi, theta, r)
    return [X, Y, Z]

def drawPoint(img, h, w, point_size):
    h = np.arange(h-int(point_size/2),h+int(point_size/2), 1)
    w = np.arange(w-int(point_size/2),w+int(point_size/2), 1)
    for i in range(len(h)):
        if(h[i]<0):
            h[i] = img.shape[0]+h[i]
        if(w[i]<0):
            w[i] = img.shape[1]+w[i]
    hs, ws = np.meshgrid(h, w)
    img[hs, ws, 0] = 1
    img[hs, ws, 1] = 0
    img[hs, ws, 2] = 0
    return img

def undistortion(img, p, size_factor):
    # Templete grid XYZ coorditnate
    arr = np.arange(-0.12, 0.12, 0.0001)*size_factor
    Y, Z = np.meshgrid(arr, -arr)
    R = np.ones(Y.shape)
    X = np.sqrt(R-np.power(Y, 2)-np.power(Z,2))

    # Set a point
    h = p[0]
    w = p[1]
    [phi, theta] = cvtPix2Sph(h, w, height, width)

    # Determine rotaion matrices
    c, s = np.cos(-(theta-np.pi/2)), np.sin(-(theta-np.pi/2))
    Ry = np.array(((c, 0, -s), (0, 1,0), (s, 0, c)))
    c, s = np.cos(-phi), np.sin(-phi)
    Rz = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))
    RotMat = np.matmul(Rz,Ry)
    
    # Rotate the templete points in XYZ spcae
    XYZ = np.concatenate((X.reshape(1, -1), Y.reshape(1, -1), Z.reshape(1, -1)), axis=0)
    XYZ2 = np.matmul(RotMat.astype(float), XYZ.astype(float))
    X2_, Y2_, Z2_ = np.split(XYZ2, 3, axis=0)
    X2 = X2_.reshape(X.shape)
    Y2 = Y2_.reshape(Y.shape)
    Z2 = Z2_.reshape(Z.shape)

    # Convert XYZ to Pixel index
    phi, theta = cvtXYZ2Pix(X2, Y2, Z2, height, width)
    theta = theta.astype(int)
    phi = phi.astype(int)
    
    # Return the undistorted patch image
    return img[theta, phi, :]

def imwrite(img, saving_path):
    if(img.dtype == 'uint8'):
        cv2.imwrite(saving_path, img[:,:,::-1])
    else:
        cv2.imwrite(saving_path, img[:,:,::-1]*255)

def imresize(img, target_size):
    img = cv2.resize(img, (target_size[0], target_size[1]))
    return img

def mkdir_(dir):
    import os
    try:
        os.makedirs(dir)
        return True
    except:
        pass
        return  False