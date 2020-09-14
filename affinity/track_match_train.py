# a combination of track and match
# 1. load fullres images, resize to 640**2
# 2. warmup: set random location for crop
# 3. loc-match: add attention
import os
import cv2
import sys
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
# from .libs.loader import VidListv1, VidListv2
import torch.backends.cudnn as cudnn
# import affinity.libs.transforms_multi as transforms

from affinity.model import track_match_comb as Model
from affinity.libs.loss import L1_loss
from affinity.libs.concentration_loss import ConcentrationSwitchLoss as ConcentrationLoss
from affinity.libs.train_utils import save_vis, AverageMeter, save_checkpoint, log_current, avg_iou_of_batch
from affinity.libs.utils import diff_crop  # , diff_crop_by_assembled_grid

from data.dataset import SenseflyTransTrain, SenseflyTransVal, getVHRRemoteDataRandomCropper, getVHRRemoteDataAugCropper

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./experiments/localization/affinity_trainable/expr_log.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


############################## helper functions ##############################


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes
    parser.add_argument("--encoder_dir", type=str,
                        default='affinity/weights/encoder_single_gpu.pth', help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str,
                        default='affinity/weights/decoder_single_gpu.pth', help="pretrained decoder")
    parser.add_argument('--resume', type=str, default='affinity/weights/checkpoint_latest.pth.tar',
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-c", "--savedir", type=str,
                        default="experiments/localization/affinity_trainable", help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18",
                        help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--batchsize", type=int, default=4, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256,
                        help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=1024,
                        help="full size for one frame.")

    parser.add_argument("--lr", type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument("--window_len", type=int, default=2,
                        help='number of images (2 for pair and 3 for triple)')
    parser.add_argument("--log_interval", type=int, default=10, help='')
    parser.add_argument("--save_interval", type=int,
                        default=8, help='save every x epoch')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument("--weight_decay", type=float,
                        default=0.005, help='weight decay')
    parser.add_argument("--device", type=int, default=2,
                        help="0~device_count-1 for single GPU, device_count for dataparallel.")
    parser.add_argument("--temp", type=int, default=1,
                        help="temprature for softmax.")

    parser.add_argument("--scale-modeling", dest='estimate_scale', action='store_const', const=True, default=False,
                        help="do scale modeling or not")
    parser.add_argument("--rand_aug", dest='rand_aug', action='store_const', const=True, default=False,
                        help="data augmentation for training")
    parser.add_argument("--strict_orth", dest='orth', action='store_const', const=True, default=False,
                        help="data augmentation for training")

    # set epoches
    parser.add_argument("--wepoch", type=int, default=10, help='warmup epoch')
    parser.add_argument("--nepoch", type=int, default=64, help='max epoch')

    # concenration regularization
    parser.add_argument("--lc", type=float, default=1e4,
                        help='weight of concentration loss')
    parser.add_argument("--lc_win", type=int, default=8,
                        help='win_len for concentration loss')

    # orthorganal regularization
    parser.add_argument("--color_switch", type=float,
                        default=0.1, help='weight of color switch loss')
    parser.add_argument("--coord_switch", type=float,
                        default=0.1, help='weight of color switch loss')

    print("Begin parser arguments.")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    args.savepatch = os.path.join(args.savedir, 'savepatch')
    args.logfile = open(os.path.join(args.savedir, "logargs.txt"), "w")
    args.multiGPU = args.device == torch.cuda.device_count()

    if not args.multiGPU:
        torch.cuda.set_device(args.device)
    if not os.path.exists(args.savepatch):
        os.mkdir(args.savepatch)

    args.vis = True
    if args.color_switch > 0:
        args.color_switch_flag = True
    else:
        args.color_switch_flag = False
    if args.coord_switch > 0:
        args.coord_switch_flag = True
    else:
        args.coord_switch_flag = False

    try:
        from tensorboardX import SummaryWriter
        global writer
        writer = SummaryWriter()
    except ImportError:
        args.vis = False
    print(' '.join(sys.argv))
    print('\n')
    args.logfile.write(' '.join(sys.argv))
    args.logfile.write('\n')

    for k, v in args.__dict__.items():
        print(k, ':', v)
        args.logfile.write('{}:{}\n'.format(k, v))
    args.logfile.close()
    return args


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.nepoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def create_loader(args, aug_options):
    '''dataset_train_warm = VidListv1(
        args.videoRoot, args.videoList, args.patch_size, args.rotate, args.scale)
    dataset_train = VidListv2(args.videoRoot, args.videoList, args.patch_size,
                              args.window_len, args.rotate, args.scale, args.full_size)'''
    # dataset_train_warm = SenseflyTransTrain(crop_size=args.patch_size)
    # dataset_train = SenseflyTransTrain(crop_size=args.patch_size)
    if args.rand_aug:
        dataset_train_warm, dataset_train, dataset_val = getVHRRemoteDataAugCropper(aug=aug_options)
    else:
        dataset_train_warm, dataset_train, dataset_val = getVHRRemoteDataRandomCropper()
    if args.multiGPU:
        train_loader_warm = torch.utils.data.DataLoader(
            dataset_train_warm, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)
    else:
        train_loader_warm = torch.utils.data.DataLoader(
            dataset_train_warm, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)
    return train_loader_warm, train_loader


def train(args, aug_ops):
    loader_warm, loader = create_loader(args, aug_ops)
    cudnn.benchmark = True
    best_loss = 1e10
    start_epoch = 0

    model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp,
                  Resnet=args.Resnet, color_switch=args.color_switch_flag, coord_switch=args.coord_switch_flag,
                  model_scale=args.estimate_scale)
    if args.multiGPU:
        model = torch.nn.DataParallel(model).cuda()
        closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
                                  F_size=torch.Size((args.batchsize // torch.cuda.device_count(), 2,
                                                     args.patch_size // 8, args.patch_size // 8)), temp=args.temp)
        closs = nn.DataParallel(closs).cuda()
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, model._modules['module'].parameters()), args.lr)
    else:
        closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
                                  F_size=torch.Size((args.batchsize, 2,
                                                     args.patch_size // 8,
                                                     args.patch_size // 8)), temp=args.temp)
        model.cuda()
        closs.cuda()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss'] * 1e5  # in order to do transfer training
            if not args.multiGPU:
                model = torch.nn.DataParallel(model).cuda()
                model.load_state_dict(checkpoint['state_dict'])
                model = model.module
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{} ({})' (epoch {})"
                  .format(args.resume, best_loss, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.nepoch):
        if epoch < args.wepoch:
            lr = adjust_learning_rate(args, optimizer, epoch)
            print("Base lr for epoch {}: {}.".format(
                epoch, optimizer.param_groups[0]['lr']))
            best_loss = train_iter(
                args, loader_warm, model, closs, optimizer, epoch, best_loss)
        else:
            lr = adjust_learning_rate(args, optimizer, epoch - args.wepoch)
            print("Base lr for epoch {}: {}.".format(
                epoch, optimizer.param_groups[0]['lr']))
            best_loss = train_iter(args, loader, model,
                                   closs, optimizer, epoch, best_loss)


def forward(frame1, frame2, model, warm_up, args, patch_size=None):
    n, c, h, w = frame1.size()
    if warm_up:
        output = model(frame1, frame2, patch_size=[patch_size // 8, patch_size // 8], strict_orth=args.orth)
    else:
        output = model(frame1, frame2, warm_up=False,
                       patch_size=[patch_size // 8, patch_size // 8], strict_orth=args.orth)
        bbox = output[2][0]
        # gt patch
        # print("HERE2: ", frame2.size(), new_c, patch_size)
        color2_gt = diff_crop(frame2, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                              patch_size, patch_size)
        # color2_gt = diff_crop_by_assembled_grid(frame2, new_c[:, :2], new_c[:, 2:] - 1)
        output.append(color2_gt)
    return output


def train_iter(args, loader, model, closs, optimizer, epoch, best_loss):
    losses = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    c_losses = AverageMeter()

    iou_counter = AverageMeter()

    model.train()
    end = time.time()
    if args.coord_switch_flag:
        coord_switch_loss = nn.L1Loss()
        sc_losses = AverageMeter()

    if epoch < 1 or (args.wepoch <= epoch < args.wepoch + 2):
        thr = None
    else:
        thr = 2.5

    for i, frames in enumerate(loader):
        frame1_var = frames[0].cuda()
        frame2_var = frames[1].cuda()

        if epoch < args.wepoch:
            output = forward(frame1_var, frame2_var, model, warm_up=True, args=args, patch_size=args.patch_size)
            img_size = frame1_var.size(2)
            if img_size > args.patch_size:
                center = img_size // 2
                half_patch_size = args.patch_size // 2
                frame1_var = frame1_var[..., center - half_patch_size:center - half_patch_size + args.patch_size,
                             center - half_patch_size:center - half_patch_size + args.patch_size]
                frame2_var = frame2_var[..., center - half_patch_size:center - half_patch_size + args.patch_size,
                             center - half_patch_size:center - half_patch_size + args.patch_size]
            color2_est = output[0]
            aff = output[1]
            b, x, _ = aff.size()
            color1_est = None
            if args.color_switch_flag:
                color1_est = output[2]
            loss_ = L1_loss(color2_est, frame2_var, 10, 10,
                            thr=thr, pred1=color1_est, frame1_var=frame1_var)

            if epoch >= 1 and args.lc > 0:
                constraint_loss = torch.sum(
                    closs(aff.view(b, 1, x, x))) * args.lc
                c_losses.update(constraint_loss.item(), frame1_var.size(0))
                loss = loss_ + constraint_loss
            else:
                loss = loss_
            if (i % args.log_interval == 0):
                save_vis(i, color2_est, frame2_var, frame1_var,
                         frame2_var, os.path.join(args.savepatch, 'warm', str(epoch)))
        else:
            corners_gt = frames[-1]
            output = forward(frame1_var, frame2_var, model, args=args,
                             warm_up=False, patch_size=args.patch_size)
            img_size = frame1_var.size(2)
            if img_size > args.patch_size:
                center = img_size // 2
                half_patch_size = args.patch_size // 2
                frame1_var = frame1_var[..., center - half_patch_size:center - half_patch_size + args.patch_size,
                             center - half_patch_size:center - half_patch_size + args.patch_size]
            color2_est = output[0]
            aff = output[1]
            new_c, modeled_bbox = output[2]
            coords = output[3]
            # f1_grid, fcrop_grid = output[3]
            # Fcolor2_crop = output[-1]
            color2_crop = output[-1]

            b, x, x = aff.size()
            color1_est = None
            count = 3

            constraint_loss = torch.sum(closs(aff.view(b, 1, x, x))) * args.lc
            c_losses.update(constraint_loss.item(), frame1_var.size(0))

            avg_iou = avg_iou_of_batch(modeled_bbox, corners_gt)
            iou_counter.update(avg_iou, modeled_bbox.size(0))

            if args.color_switch_flag:
                count += 1
                color1_est = output[count]

            loss_color = L1_loss(color2_est, color2_crop, 10, 10,
                                 thr=thr, pred1=color1_est, frame1_var=frame1_var)
            loss_ = loss_color + constraint_loss

            if args.coord_switch_flag:
                count += 1
                grids = output[count]
                C11 = output[count + 1]
                loss_coord = args.coord_switch * coord_switch_loss(C11, grids)
                loss = loss_ + loss_coord
                sc_losses.update(loss_coord.item(), frame1_var.size(0))
            else:
                loss = loss_

            if (i % args.log_interval == 0):
                save_vis(i, color2_est, color2_crop, frame1_var,
                         frame2_var, os.path.join(args.savepatch, 'train', str(epoch)),
                         coords, corners_gt, modeled_bbox)

        losses.update(loss.item(), frame1_var.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch >= args.wepoch and args.coord_switch_flag:
            if i % args.log_interval == 0:
                info_str = 'Epoch: [{0}][{1}/{2}]\t' \
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                           'Color Loss {loss.val:.4f} ({loss.avg:.4f})\t ' \
                           'Coord switch Loss {scloss.val:.4f} ({scloss.avg:.4f})\t ' \
                           'Constraint Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t ' \
                           'IOU Average {iou_meter.val:.4f}({iou_meter.avg:.4f})\t '.format(
                    epoch, i + 1, len(loader), batch_time=batch_time, loss=losses, scloss=sc_losses, c_loss=c_losses,
                    iou_meter=iou_counter)
                logger.info(info_str)
                print(info_str)
        else:
            if i % args.log_interval == 0:
                info_str = 'Epoch: [{0}][{1}/{2}]\t' \
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                           'Color Loss {loss.val:.4f} ({loss.avg:.4f})\t ' \
                           'Constraint Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t '.format(
                    epoch, i + 1, len(loader), batch_time=batch_time, loss=losses, c_loss=c_losses)
                logger.info(info_str)
                print(info_str)

        if ((i + 1) % args.save_interval == 0):
            is_best = losses.avg < best_loss
            best_loss = min(losses.avg, best_loss)
            checkpoint_path = os.path.join(
                args.savedir, 'vhr_orig_checkpoint_latest.pth.tar')
            if not args.multiGPU:
                save_model = torch.nn.DataParallel(model).cuda()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': save_model.state_dict(),
                'best_loss': best_loss,
            }, is_best, filename=checkpoint_path, savedir=args.savedir)
            log_current(epoch, losses.avg, best_loss,
                        filename="log_current.txt", savedir=args.savedir)

    return best_loss


if __name__ == '__main__':
    args = parse_args()
    origin_save_dir = args.savepatch
    args.orth = True
    expr_1_dir = os.path.join(origin_save_dir, 'orth', 'scale_only')
    aug_1 = {'scale': 2}
    expr_2_dir = os.path.join(origin_save_dir, 'orth', 'rot_only')
    aug_2 = {'rotate': 100}
    expr_3_dir = os.path.join(origin_save_dir, 'orth', 'aug_light')
    aug_3 = {'scale': 1.5, 'rotate': 60}
    expr_4_dir = os.path.join(origin_save_dir, 'orth', 'aug_mid')
    aug_4 = {'scale': 2, 'rotate': 100, 'erase': (0.5, 0.01, 0.02, 0.6)}

    args.savepatch = expr_1_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_1 warm')
    train(args, aug_1)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_1 nowarm')
    train(args, aug_1)

    args.savepatch = expr_2_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_2 warm')
    train(args, aug_2)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_2 nowarm')
    train(args, aug_2)

    args.savepatch = expr_3_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_3 warm')
    train(args, aug_3)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_3 nowarm')
    train(args, aug_3)

    args.savepatch = expr_4_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_4 warm')
    train(args, aug_4)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_4 nowarm')
    train(args, aug_4)

    args.orth = False
    expr_1_dir = os.path.join(origin_save_dir, 'norm', 'scale_only')
    aug_1 = {'scale': 2}
    expr_2_dir = os.path.join(origin_save_dir, 'norm', 'rot_only')
    aug_2 = {'rotate': 100}
    expr_3_dir = os.path.join(origin_save_dir, 'norm', 'aug_light')
    aug_3 = {'scale': 1.5, 'rotate': 60}
    expr_4_dir = os.path.join(origin_save_dir, 'norm', 'aug_mid')
    aug_4 = {'scale': 2, 'rotate': 100, 'erase': (0.5, 0.01, 0.02, 0.6)}

    args.savepatch = expr_1_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_1 warm')
    train(args, aug_1)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_1 nowarm')
    train(args, aug_1)

    args.savepatch = expr_2_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_2 warm')
    train(args, aug_2)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_2 nowarm')
    train(args, aug_2)

    args.savepatch = expr_3_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_3 warm')
    train(args, aug_3)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_3 nowarm')
    train(args, aug_3)

    args.savepatch = expr_4_dir
    args.wepoch = 20
    args.nepoch = 32
    logger.info('expr_4 warm')
    train(args, aug_4)
    args.wepoch = 10
    args.nepoch = 24
    args.savepatch = os.path.join(args.savepatch, 'no_warm')
    logger.info('expr_4 nowarm')
    train(args, aug_4)

    # writer.close()
