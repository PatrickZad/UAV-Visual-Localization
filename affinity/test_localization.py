# OS libraries
import os
import argparse
import sys

# Pytorch
import torch
import torch.nn as nn

# Customized libraries
from affinity.libs.test_utils import *
from affinity.libs.model import transform
from affinity.libs.utils import norm_mask

from affinity.model import track_match_comb as Model


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes

    parser.add_argument("--encoder_dir", type=str, default='affinity_t_lib/weights/encoder_single_gpu.pth',
                        help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str, default='affinity_t_lib/weights/decoder_single_gpu.pth',
                        help="pretrained decoder")
    parser.add_argument('--resume', type=str, default='affinity_t_lib/weights/checkpoint_latest.pth.tar',
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-c", "--savedir", type=str, default="match_track_comb/", help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)

    parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=1024, help="full size for one frame.")
    parser.add_argument("--window_len", type=int, default=2, help='number of images (2 for pair and 3 for triple)')
    parser.add_argument("--device", type=int, default=4,
                        help="0~device_count-1 for single GPU, device_count for dataparallel.")
    parser.add_argument("--temp", type=int, default=1, help="temprature for softmax.")

    print("Begin parser arguments.")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    args.savepatch = os.path.join(args.savedir, 'savepatch')
    if not os.path.exists(args.savepatch):
        os.mkdir(args.savepatch)
    return args


if (__name__ == '__main__'):
    from data.dataset import SenseflyTransVal, VHRRemoteVal, getVHRRemoteDataRandomCropper
    from torch.utils.data import DataLoader
    import cv2
    import numpy as np


    def flat2coord(idx, length, samp=8):
        y = idx // length * samp + samp // 2
        x = idx % length * samp + samp // 2
        return x, y


    def cv_point(pt, orientation=0):
        point = cv2.KeyPoint()
        point.size = 17
        point.angle = orientation
        point.class_id = -1
        point.octave = 0
        point.response = 0
        point.pt = (pt[0], pt[1])
        return point


    def cv_match(qidx, tidx, dist=0., img_idx=0):
        match = cv2.DMatch(qidx, tidx, img_idx, dist)
        return match


    args = parse_args()
    # loading pretrained model
    model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp, Resnet=args.Resnet,
                  color_switch=False, coord_switch=False)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.cuda()
    model.eval()

    # start testing
    dataset_t, dataset = getVHRRemoteDataRandomCropper(0, map_size=args.full_size)
    # scale_f = 0.4
    # dataset = SenseflyTransVal(scale_f=scale_f)
    loader = DataLoader(dataset,
                        batch_size=1)  # batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True,
    # drop_last=True)
    # dataset_dir = dataset.get_dataset_dir()
    save_dir = args.savedir
    for env_dir, img_t, img_file, map_t, map_file, (img_arr, map_arr) in loader:
        img_t = img_t.cuda()
        map_t = map_t.cuda()
        img_arr = img_arr.squeeze().numpy().astype(np.uint8)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        map_arr = map_arr.squeeze().numpy().astype(np.uint8)
        map_arr = cv2.cvtColor(map_arr, cv2.COLOR_RGB2BGR)

        '''img_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'imgs', img_file[0]))
        img_arr = cv2.resize(img_arr, (0, 0), fx=scale_f, fy=scale_f)
        map_arr = cv2.imread(os.path.join(dataset_dir, env_dir[0], 'map', map_file[0]))
        map_arr = cv2.resize(map_arr, (0, 0), fx=scale_f, fy=scale_f)'''
        loc_box, aff = model(img_t, map_t, False, patch_size=[args.patch_size // 8, args.patch_size // 8],
                             test_result=True)
        f_patch_size = args.patch_size // 8
        f_ref_size = args.full_size // 8
        img_coords = [cv_point(flat2coord(idx, f_patch_size)) for idx in range(f_patch_size ** 2)]
        aff = aff.squeeze()
        corres_ids = np.argmax(aff, axis=-1)
        corres_coords = [cv_point(flat2coord(corres_coords[i], f_ref_size)) for i in range(corres_ids.shape[0])]
        cv_matches = [cv_match(i, i) for i in range(len(img_coords))]
        loc_box = loc_box.squeeze()
        aff_arr = (aff * 255).cpu().detach().numpy()
        aff_arr = aff_arr.astype(np.uint8)
        img_size = img_arr.shape[0]
        if img_size > args.patch_size:
            center = img_size // 2
            half_patch_size = args.patch_size // 2
            img_arr = img_arr[..., center - half_patch_size:center - half_patch_size + args.patch_size[0],
                      center - half_patch_size:center - half_patch_size + args.patch_size[0]]

        heat_map = cv2.applyColorMap(aff_arr, cv2.COLORMAP_JET)
        pts = np.array(
            [[[loc_box[0], loc_box[1]], [loc_box[2], loc_box[1]], [loc_box[2], loc_box[3]], [loc_box[0], loc_box[3]]]],
            np.int32)
        cv2.polylines(map_arr, pts, True, (0, 0, 255), thickness=3)
        background = np.zeros((max(img_arr.shape[0], map_arr.shape[0]), img_arr.shape[1] + map_arr.shape[1], 3))
        background[:img_arr.shape[0], :img_arr.shape[1], :] = img_arr
        background[:map_arr.shape[0], img_arr.shape[1]:img_arr.shape[1] + map_arr.shape[1], :] = map_arr
        cv2.imwrite(os.path.join(save_dir, env_dir[0] + '_' + img_file[0][:-4] + map_file[0]), background)
        cv2.imwrite(os.path.join(save_dir, env_dir[0] + '_' + img_file[0][:-4] + map_file[0][:-4] + 'aff.jpg'),
                    heat_map)

        match_result = np.zeros((max(img_arr.shape[0], map_arr.shape[0]), img_arr.shape[1] + map_arr.shape[1], 3))
        cv2.drawMatches(img_arr, img_coords, map_arr, corres_coords,
                        cv_matches, match_result, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(os.path.join(save_dir, env_dir[0] + '_' + img_file[0][:-4] + map_file[0][:-4] + 'match.jpg'),
                    match_result)
