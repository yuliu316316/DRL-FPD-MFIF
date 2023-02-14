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

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

def Fusion(models, inputdata1, inputdata2, inputdata3):
    x1 = models.layer1(inputdata1)
    x1 = models.layer2(x1)
    x1 = models.layer3(x1)
    x1 = models.layer4(x1)
    x2 = models.layer1(inputdata2)
    x2 = models.layer2(x2)
    x2 = models.layer3(x2)
    x2 = models.layer4(x2)
    x3 = models.layer1(inputdata3)
    x3 = models.layer2(x3)
    x3 = models.layer3(x3)
    x3 = models.layer4(x3)
    mid_tensor = torch.max(x1, x2)
    max_tensor = torch.max(mid_tensor, x3)
    y = models.layer5(max_tensor)
    y = models.layer6(y)
    y = models.layer7(y)
    y = models.layer8(y)
    y = models.sigmoid(y)
    # x4 = torch.cat((x1,x2,x3),1)
    return y

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
    img = img.mul(255).byte()     # img *= 255
    img = img.detach().cpu().numpy()
    img = np.transpose(np.squeeze(img, axis=0), (1, 2, 0))
    # img = np.squeeze(img)     # works if grayscale
    # maxValue = img.max()
    # img = img * 255 / maxValue
    # img = img.astype(np.uint8)    # img: float32->uint8
    return img


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


def compare(f1, f2, f3):
    device = f1.device
    index_map = torch.zeros(f1.shape).to(device)
    mid_term = torch.min(f1, f2)
    min = torch.min(mid_term, f3)
    h, w = f1.shape
    for i in range(h):
        for j in range(w):
            if f1[i, j] == min[i, j]:
                index_map[i, j] = 0
            elif f2[i, j] == min[i, j]:
                index_map[i, j] = 1
            else:
                index_map[i, j] = 2
    index_map = index_map.cpu().numpy().astype(np.int)
    # device = f1.device
    # weight_zeros = torch.zeros(f1.shape).to(device)
    # weight_ones = torch.ones(f1.shape).to(device)
    # # get decision map
    # dm_tensor = torch.where(f1 < f2, weight_ones, weight_zeros).to(device)
    # dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
    return index_map


def toimg(img):
    maxValue = img.max()
    img255 = img * 255 / maxValue
    img255 = img255.astype(np.uint8)  # img: float32->uint8
    return img255


def remove_connect(dm, transfor):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(transfor.astype(np.uint8), connectivity=8)
    all_lines = np.where(stats[:, 4] < 0.015 * h * w)
    if any(all_lines[0]):
        for line in all_lines[0]:
            bound = labels[stats[line, 1] - 1:stats[line, 1] + stats[line, 3] + 1,
                    stats[line, 0] - 1:stats[line, 0] + stats[line, 2] + 1]
            list1 = np.where(bound == line)
            list2 = np.where(bound != line)
            # print(len(list1[0]))
            L = len(list1[0])
            R = len(list2[0])
            distance = np.zeros([L, R])
            for LL in range(L):
                for RR in range(R):
                    x = np.array((list1[0][LL], list1[1][LL]))
                    y = np.array((list2[0][RR], list2[1][RR]))
                    distance[LL, RR] = np.sqrt(np.sum(np.square(x - y)))
                index = np.where(distance[LL, :] == distance[LL, :].min())
                dm[stats[line, 1] - 1 + list1[0][LL], stats[line, 0] - 1 + list1[1][LL]] \
                    = dm[stats[line, 1] - 1 + list2[0][index[0][0]], stats[line, 0] - 1 + list2[1][index[0][0]]]
    else:
        dm=dm
    return dm


num = 1
win_size = 7
for num in range(1, 5):
    original_path1 = '/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/sourceimages/multiple/lytro-{}{}-A.tif'.format(num // 10, num % 10)
    original_path2 = '/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/sourceimages/multiple/lytro-{}{}-B.tif'.format(num // 10, num % 10)
    original_path3 = '/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/sourceimages/multiple/lytro-{}{}-C.tif'.format(num // 10, num % 10)

    tic = time.time()
    img1_org = Image.open(original_path1).convert('RGB')
    img2_org = Image.open(original_path2).convert('RGB')
    img3_org = Image.open(original_path3).convert('RGB')
    tran = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))])
    img1_org = tran(img1_org)
    img2_org = tran(img2_org)
    img3_org = tran(img3_org)
    img1_org = img1_org.unsqueeze(0).cuda()
    img2_org = img2_org.unsqueeze(0).cuda()
    img3_org = img3_org.unsqueeze(0).cuda()
    model.eval()
    img_fused = Fusion(model, img1_org, img2_org, img3_org)
    im = mpimg.imread(original_path1)
    FUSE = post_process1(img_fused)

    img1_org = Image.open(original_path1).convert('RGB')
    img2_org = Image.open(original_path2).convert('RGB')
    img3_org = Image.open(original_path3).convert('RGB')

    I1 = np.asarray(img1_org)
    I2 = np.asarray(img2_org)
    I3 = np.asarray(img3_org)
    F = np.asarray(FUSE)

    img1_org = tran(img1_org)
    img2_org = tran(img2_org)
    img3_org = tran(img3_org)
    fuse = tran(FUSE)
    img1_org = img1_org.unsqueeze(0).to(device)
    img2_org = img2_org.unsqueeze(0).to(device)
    img3_org = img3_org.unsqueeze(0).to(device)
    fuse = fuse.unsqueeze(0).to(device)
    model.eval()
    F1 = resnet_multi(model, img1_org)
    F2 = resnet_multi(model, img2_org)
    F3 = resnet_multi(model, img3_org)
    Fuse = resnet_multi(model, fuse)

    mse1 = (Fuse - F1)**2    # 1*32*h*w
    mse2 = (Fuse - F2)**2
    mse3 = (Fuse - F3)**2
    mse1 = torch.sum(mse1, dim=1).unsqueeze(0).to(device)
    mse2 = torch.sum(mse2, dim=1).unsqueeze(0).to(device)
    mse3 = torch.sum(mse3, dim=1).unsqueeze(0).to(device)
    mse1 = mean_filter(mse1, win_size).squeeze()
    mse2 = mean_filter(mse2, win_size).squeeze()
    mse3 = mean_filter(mse3, win_size).squeeze()
    first_dm = compare(mse1, mse2, mse3)
    first_dm255 = toimg(first_dm)

    # Morphology filter and Small region removal
    h, w = I1.shape[:-1]
    se = skimage.morphology.disk(5)  # 'disk' kernel with ks size for structural element
    dm = first_dm
    dm_transfor1 = np.where(dm == 0, 1, 0)
    dm1 = morphology.remove_small_holes(dm_transfor1 == 1, 0.015 * h * w)
    diference1 = dm1-dm_transfor1
    index1 = np.where(diference1 == 1)
    dm[index1] = 0

    dm_transfor2 = np.where(dm == 1, 1, 0)
    dm2 = morphology.remove_small_holes(dm_transfor2 == 1, 0.015 * h * w)    # remove small holes in 1
    diference2 = dm2-dm_transfor2
    index2 = np.where(diference2 == 1)
    dm[index2] = 1

    dm_transfor3 = np.where(dm == 2, 1, 0)
    dm3 = morphology.remove_small_holes(dm_transfor3 == 1, 0.015 * h * w)
    diference3 = dm3-dm_transfor3
    index3 = np.where(diference3 == 1)
    dm[index3] = 2
    second_dm = toimg(dm)

    decision_map0 = dm
    m = 9
    pad = m // 2
    FF0 = np.zeros(I1.shape)
    core0 = np.ones([m, m])
    Z_sqr0 = convolve2d(decision_map0, core0, mode="same", boundary="symm")
    boundary_map0 = np.zeros(Z_sqr0.shape)
    for i in range(h):
        for j in range(w):
            Zp = decision_map0[i, j]
            Zb = Z_sqr0[i, j]
            if Zp == 0 and Zb == 0:
                FF0[i,j,:] = I1[i,j,:]
                boundary_map0[i,j] = Zp
            elif Zp == 1 and Zb == m ** 2:
                FF0[i,j,:] = I2[i,j,:]
                boundary_map0[i, j] = Zp
            elif Zp == 2 and Zb == 2 * (m ** 2):
                FF0[i, j, :] = I3[i, j, :]
                boundary_map0[i, j] = Zp
            else:
                FF0[i,j,:] = F[i,j,:]
                boundary_map0[i, j] = 1.5
    FF0 = FF0.astype(np.uint8)

    dm_transfor4 = np.where(dm == 0, 1, 0)
    dm = remove_connect(dm, dm_transfor4)
    dm_transfor5 = np.where(dm == 1, 1, 0)
    dm = remove_connect(dm, dm_transfor5)
    dm_transfor6 = np.where(dm == 2, 1, 0)
    dm = remove_connect(dm, dm_transfor6)

    decision_map = dm
    m = 9
    pad = m // 2
    FF = np.zeros(I1.shape)
    core = np.ones([m, m])  # m=9
    Z_sqr = convolve2d(decision_map, core, mode="same", boundary="symm")
    boundary_map = np.zeros(Z_sqr.shape)
    for i in range(h):
        for j in range(w):
            Zp = decision_map[i, j]
            Zb = Z_sqr[i, j]
            if Zp == 0 and Zb == 0:
                FF[i,j,:] = I1[i,j,:]
                boundary_map[i,j] = Zp
            elif Zp == 1 and Zb == m ** 2:
                FF[i,j,:] = I2[i,j,:]
                boundary_map[i, j] = Zp
            elif Zp == 2 and Zb == 2 * (m ** 2):
                FF[i, j, :] = I3[i, j, :]
                boundary_map[i, j] = Zp
            else:
                FF[i,j,:] = F[i,j,:]
                boundary_map[i, j] = 1.5
    FF = FF.astype(np.uint8)
    imageio.imwrite('/media/gdlls/My Book/Root/QZZ/code/DRL-FPD/results_multiple3/lytro_{}{}.tif'.format(num//10, num % 10), FF)
    toc = time.time()
    print('end {}{}'.format(num // 10, num % 10) + ',running time:{}'.format(toc - tic))
