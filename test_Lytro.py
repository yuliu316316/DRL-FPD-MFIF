import time
from PIL import Image
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import imageio
import joblib
import matplotlib.image as mpimg
import torch.nn.functional as f
import torchvision.transforms as transforms
from Networks import resnet_module3 as net
import skimage
from skimage import morphology
from scipy.signal import convolve2d

'similarity by ()^2'

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# ids = torch.cuda.device_count()
device = torch.device('cuda:0')       # CUDA:0
# device = 'cpu'

model = net.MyRegressionNet()
model_path = "/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/models/MyRegressionNet122/model.pth"
use_gpu = torch.cuda.is_available()
# use_gpu = False

if use_gpu:

    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()
    # model = model
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))   # Dataparallel:bing xing
    model.load_state_dict(torch.load(model_path))

else:

    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')
    # load params
    model.load_state_dict(state_dict)


def cov2(img, f, stride=1):
    inw, inh = img.shape
    w, h = f.shape
    outw = int((inw - w) / stride + 1)
    outh = int((inh - h) / stride + 1)
    arr = np.zeros(shape=(outw, outh))
    for g in range(outh):
        for t in range(outw):
            s = 0
            for i in range(w):
                for j in range(h):
                    s += img[i + g * stride][j + t * stride] * f[i][j]
                    # s = img[i][j] * f[i][j]
            arr[g][t] = s
    return arr


def box_filter(imgSrc, r):
        """
        Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
        :param imgSrc: np.array, image
        :param r: int, radius
        :return: imDst: np.array. result of calculation
        """
        if imgSrc.ndim == 2:
            h, w = imgSrc.shape[:2]
            imDst = np.zeros(imgSrc.shape[:2])

            # cumulative sum over h axis
            imCum = np.cumsum(imgSrc, axis=0)
            # difference over h axis
            imDst[0: r+1] = imCum[r: 2 * r+1]
            imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
            imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

            # cumulative sum over w axis
            imCum = np.cumsum(imDst, axis=1)

            # difference over w axis
            imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
            imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
            imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
                                 imCum[:, w - 2 * r - 1: w - r - 1]
        else:
            h, w = imgSrc.shape[:2]
            imDst = np.zeros(imgSrc.shape)

            # cumulative sum over h axis
            imCum = np.cumsum(imgSrc, axis=0)
            # difference over h axis
            imDst[0: r + 1] = imCum[r: 2 * r + 1]
            imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
            imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

            # cumulative sum over w axis
            imCum = np.cumsum(imDst, axis=1)

            # difference over w axis
            imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
            imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
            imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
                                 imCum[:, w - 2 * r - 1: w - r - 1]
        return imDst


def guided_filter(I, p, r, eps=0.1):
        """
        Guided Filter
        :param I: np.array, guided image
        :param p: np.array, input image
        :param r: int, radius
        :param eps: float
        :return: np.array, filter result
        """
        h, w = I.shape[:2]
        if I.ndim == 2:
            N = box_filter(np.ones((h, w)), r)
        else:
            N = box_filter(np.ones((h, w, 1)), r)
        mean_I = box_filter(I, r) / N
        mean_p = box_filter(p, r) / N
        mean_Ip = box_filter(I * p, r) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = box_filter(I * I, r) / N
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)

        if I.ndim == 2:
            b = mean_p - a * mean_I
            mean_a = box_filter(a, r) / N
            mean_b = box_filter(b, r) / N
            q = mean_a * I + mean_b
        else:
            b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
            mean_a = box_filter(a, r) / N
            mean_b = box_filter(b, r) / N
            q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
        return q


def resnet_multi(models, inputdata):
    x = models.layer1(inputdata)
    x = models.layer2(x)
    x = models.layer3(x)
    x = models.layer4(x)
    # x4 = torch.cat((x1,x2,x3),1)
    return x


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val
    return kernel


def mean_filter(matrix, kernel_size):
    kernel = torch.ones([kernel_size, kernel_size])/(kernel_size**2)
    kernel = kernel.unsqueeze(0).repeat(1, 1, 1, 1).to(device)
    result = f.conv2d(matrix, kernel, padding=kernel_size // 2)
    return result


def gauss_filter(matrix, kernel_size, sigma=0):
    kernel = gauss(kernel_size, sigma)
    kernel = torch.tensor(kernel).float().unsqueeze(0).repeat(1, 1, 1, 1)
    result = f.conv2d(matrix, kernel, padding=kernel_size // 2)
    return result


def compare(f1, f2):
    device = f1.device
    weight_zeros = torch.zeros(f1.shape).to(device)
    weight_ones = torch.ones(f1.shape).to(device)

    # get decision map
    dm_tensor = torch.where(f1 < f2, weight_ones, weight_zeros).to(device)
    dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
    return dm_np

def post_process1(img):
    img = img.mul(255).byte()     # img *= 255
    img = img.detach().cpu().numpy()
    img = np.transpose(np.squeeze(img, axis=0), (1, 2, 0))
    # img = np.squeeze(img)     # works if grayscale
    # maxValue = img.max()
    # img = img * 255 / maxValue
    # img = img.astype(np.uint8)    # img: float32->uint8
    return img

def post_process(img):
    img = img.detach().cpu().numpy()
    maxValue = img.max()
    minValue = img.min()
    img = (img -minValue) / (maxValue -minValue)
    # img = img.astype(np.uint8)    # img: float32->uint8
    return img

# num = 1
win_size = 7
for num in range(1, 21):
    original_path1 = '/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/sourceimages/Lytro/c_{}{}_1.tif'.format(num//10, num % 10)
    original_path2 = '/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/sourceimages/Lytro/c_{}{}_2.tif'.format(num//10, num % 10)

    tic = time.time()
    img1_org = Image.open(original_path1).convert('RGB')
    img2_org = Image.open(original_path2).convert('RGB')
    tran = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])
    img1_org = tran(img1_org)
    img2_org = tran(img2_org)
    img1_org = img1_org.unsqueeze(0).cuda()
    img2_org = img2_org.unsqueeze(0).cuda()
    model.eval()
    img_fused = model(img1_org, img2_org)
    im = mpimg.imread(original_path1)
    FUSE = post_process1(img_fused)

    img1_org = Image.open(original_path1).convert('RGB')
    img2_org = Image.open(original_path2).convert('RGB')
    I1 = np.asarray(img1_org)
    I2 = np.asarray(img2_org)
    F = np.asarray(FUSE)

    img1_org = tran(img1_org)
    img2_org = tran(img2_org)
    fuse = tran(FUSE)
    img1_org = img1_org.unsqueeze(0).to(device)
    img2_org = img2_org.unsqueeze(0).to(device)
    fuse = fuse.unsqueeze(0).to(device)
    model.eval()
    F1 = resnet_multi(model, img1_org)
    F2 = resnet_multi(model, img2_org)
    Fuse = resnet_multi(model, fuse)

    mse1 = (Fuse - F1)**2    # 1*32*h*w   (||F-A||_2)^2
    mse2 = (Fuse - F2)**2
    mse1 = torch.sum(mse1, dim=1).unsqueeze(0).to(device)
    mse2 = torch.sum(mse2, dim=1).unsqueeze(0).to(device)
    mse1 = mean_filter(mse1, win_size).squeeze()
    mse2 = mean_filter(mse2, win_size).squeeze()
    mse1to1 = post_process(mse1)
    mse2to1 = post_process(mse2)
    first_dm = compare(mse1, mse2)

    # Morphology filter and Small region removal
    h, w = I1.shape[:-1]
    se = skimage.morphology.disk(5)  # 'disk' kernel with ks size for structural element
    dm = first_dm

    dm = morphology.remove_small_holes(dm == 0, 0.015 * h * w)
    dm = np.where(dm, 0, 1)

    dm = morphology.remove_small_holes(dm == 1, 0.015 * h * w)
    dm = np.where(dm, 1, 0)
    m = 9
    FF = np.zeros(I1.shape)
    core = np.ones([m, m])
    Z_sqr = convolve2d(dm, core, mode="same", boundary="symm")
    decision_map = np.zeros(Z_sqr.shape)
    for i in range(h):
        for j in range(w):
            Zp = dm[i, j]
            Zb = Z_sqr[i, j]
            if Zp == 1 and Zb == m**2:
                FF[i,j,:] = I1[i,j,:]
                decision_map[i,j] = Zp
            elif Zp == 0 and Zb == 0:
                FF[i,j,:] = I2[i,j,:]
                decision_map[i, j] = Zp
            else:
                FF[i,j,:] = F[i,j,:]
                decision_map[i, j] = 0.5
    FF = FF.astype(np.uint8)
    imageio.imwrite('/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/results_lytro/c_{}{}.tif'.format(num//10, num % 10), FF)
    toc = time.time()
    print('end {}{}'.format(num//10, num % 10)+',running time:{}'.format(toc-tic))
