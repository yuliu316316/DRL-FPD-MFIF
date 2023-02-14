import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from PIL import Image
import PIL.ImageOps
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from Networks import resnet_module_noscale2 as net
from Networks.pytorch_msssim import ssim, msssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')        # CUDA:1
# device_ids = range(torch.cuda.device_count())  # torch.cuda.device_count()=2
# print(len(device_ids))
use_gpu = torch.cuda.is_available()

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # sourceA B中任选一个
        if img0_tuple[1] == 0:   # if it is in sourceA
            img1_tuple = img0_tuple[0][0:-20]+'B'+img0_tuple[0][-19:]
        else:
            img1_tuple = img0_tuple[0][0:-20] + 'A' + img0_tuple[0][-19:]
        truth_tuple = '/media/wanglei/_data2/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/groundtruth'+img0_tuple[0][-19:-7] + img0_tuple[0][-4:]
        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple).convert('RGB')
        truth = Image.open(truth_tuple).convert('RGB')

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            truth = PIL.ImageOps.invert(truth)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            tran = transforms.ToTensor()
            truth = tran(truth)
            # print(torch.min(truth))

        return img0, img1, truth

    def __len__(self):
        return len(self.imageFolderDataset.imgs)  # (path , label)


def parse_args():                 # 参数解析器
    parser = argparse.ArgumentParser()
    # 增加属性
    parser.add_argument('--name', default='MyRegressionNet', help='model name: (default: arch+timestamp)')
    # parser.add_argument('--dataset', default='cifar100',
    #                     choices=['cifar100', 'imagenet'],
    #                     help='dataset name')
    # parser.add_argument('--imagenet-dir', help='path to ImageNet directory')
    # parser.add_argument('--arch', default='res2next29_6cx24wx6scale_se',
    #                     choices=resnet_module.__all__,
    #                     help='model architecture')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float)    # 5e-4
    parser.add_argument('--a', default=1.5, type=float)
    # parser.add_argument('--milestones', default='25,50', type=str)
    # MultiStepLR三段式lr，epoch进入milestones范围内即乘以gamma，离开milestones范围之后再乘以gamma
    # parser.add_argument('--gamma', default=0.1, type=float)            # MultiStepLR gamma
    parser.add_argument('--gamma', default=0.9, type=float)     # ExponentialLR gamma
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)             # 5e-4
    # parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()    # 属性给与args实例：add_argument 返回到 args 子类实例

    return args


class AverageMeter(object):
    """Computes and stores the average and current value 计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion1, criterion2, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    # acc1s = AverageMeter()
    # acc5s = AverageMeter()
    a = args.a
    model.train()

    for i, (input1, input2, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()

        output = model(input1, input2)
        loss = a * criterion2(output, target) + (1 - criterion1(output, target))   # a*L1/MSE + (1-SSIM)
        # loss = criterion2(output, target) + a * (1 - criterion1(output, target))  # L1/MSE + a*(1-SSIM)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input1.size(0))
        # acc1s.update(acc1.item(), input.size(0))
        # acc5s.update(acc5.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        # ('acc1', acc1s.avg),
        # ('acc5', acc5s.avg),
    ])

    return log


def validate(args, val_loader, model, criterion1, criterion2):
    losses = AverageMeter()
    a = args.a
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input1, input2, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

            output = model(input1, input2)
            loss = a * criterion2(output, target) + (1 - criterion1(output, target))   # a*L1/MSE + (1-SSIM)
            # loss = criterion2(output, target) + a * (1 - criterion1(output, target))  # L1/MSE + a*(1-SSIM)

            losses.update(loss.item(), input1.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log


def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)               # 创建文件夹保存模型

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))     # 打印参数配置
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:    # 写入参数文件
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    cudnn.benchmark = True

    # X_train, y_train, X_test, y_test = load_data()        # 先假定有训练集和验证集
    # 定义文件dataset
    training_dir = "/media/wanglei/_data2/VOCtrainval_320-2012/VOC2012/train/"  # 训练集地址
    folder_dataset_train = torchvision.datasets.ImageFolder(root=training_dir)
    test_dir = "/media/wanglei/_data2/VOCtrainval_320-2012/VOC2012/test/"  # 训练集地址
    folder_dataset_test = torchvision.datasets.ImageFolder(root=test_dir)

    # 定义图像dataset
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))
                                         ])
    siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train,
                                                  transform=transform_train,
                                                  should_invert=False)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transform_test,
                                                 should_invert=False)

    # 定义图像dataloader
    train_loader = DataLoader(siamese_dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)

    test_loader = DataLoader(siamese_dataset_test,
                             shuffle=True,
                             batch_size=args.batch_size)
    model = net.MyRegressionNet()
    if use_gpu:
        model = model.cuda()   # 导入模型
        model.cuda()
        # if len(device_ids) > 1:
        #     model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))  # 前提是model已经.cuda() 了
    else:
        model = model
    criterion1 = ssim
    # criterion2 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)    # Adam法优化,filter是为了固定部分参数
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)           # 梯度下降法优化
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #         milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)     # 学习率Lr调度程序
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'val_loss'])
    # best_acc = 0

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion1, criterion2, optimizer, epoch)     # 训练集
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion1, criterion2)   # 验证集

        print('loss %.4f - val_loss %.4f' %(train_log['loss'], val_log['loss']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            # train_log['acc1'],
            # train_log['acc5'],
            val_log['loss'],
            # val_log['acc1'],
            # val_log['acc5'],
        ], index=['epoch', 'lr', 'loss', 'val_loss'])                   # Series创建字典

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)        # log:训练的日志记录

        scheduler.step(epoch)  # adjust lr

        # if val_log['acc1'] > best_acc:
        #     torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
        #     best_acc = val_log['acc1']
        #     print("=> saved best model")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch) %args.name)


if __name__ == '__main__':
    main()


