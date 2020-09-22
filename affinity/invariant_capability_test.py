import cv2 as cv
from affinity.model import track_match_comb as Model
from affinity.libs.test_utils import *
import argparse
import torch.nn as nn
import torch
import os
from math import ceil
from shapely.geometry import Polygon, MultiPoint
import data.vhr_remote as vhr
from torch.utils.data import DataLoader
import logging
from affinity.libs.utils import diff_crop


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes

    parser.add_argument("--encoder_dir", type=str, default='affinity/weights/encoder_single_gpu.pth',
                        help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str, default='affinity/weights/decoder_single_gpu.pth',
                        help="pretrained decoder")
    parser.add_argument('--resume', type=str, default='affinity/weights/checkpoint_latest.pth.tar',
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-c", "--savedir", type=str, default="match_track_comb/", help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)

    parser.add_argument("--patch_size", type=int, default=384, help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=1424, help="full size for one frame.")
    parser.add_argument("--window_len", type=int, default=2, help='number of images (2 for pair and 3 for triple)')
    parser.add_argument("--device", type=int, default=0,
                        help="0~device_count-1 for single GPU, device_count for dataparallel.")
    parser.add_argument("--temp", type=int, default=1, help="temprature for softmax.")

    parser.add_argument("--scale-modeling", dest='estimate_scale', action='store_const', const=True, default=False,
                        help="do scale modeling or not")
    parser.add_argument("--rand_aug", dest='rand_aug', action='store_const', const=True, default=False,
                        help="data augmentation for training")
    parser.add_argument("--strict_orth", dest='orth', action='store_const', const=True, default=False,
                        help="data augmentation for training")
    print("Begin parser arguments.")
    args = parser.parse_args()

    return args


def draw_pt(img, pts, color=(0, 0, 255)):
    for pt in pts:
        h, w = img.shape[:2]
        left = ceil(max(0, pt[0] - 5))
        right = int(min(w, pt[0] + 5))
        top = ceil(max(0, pt[1] - 5))
        bottom = int(min(h, pt[1] + 5))
        img[top:bottom, left:right, :] = color


def draw_bbox(img, bbox):
    """
    INPUTS:
     - segmentation, h * w * 3 numpy array
     - bbox: left, top, right, bottom
    OUTPUT:
     - image with a drawn bbox
    """
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    color = np.array([51, 255, 255], dtype=np.uint8)
    c = tuple(map(int, color))
    img = cv2.rectangle(img, pt1, pt2, c, 5)
    return img


def draw_matches(img1, img2, f1_coords, f2_coords, upsamp_factor=8):
    # cv2.imwrite('./img1.jpg', img1)
    # cv2.imwrite('./img2.jpg', img2)
    img1_coords = [cv_point(fcoord2imgcoord_center(f1_coords[idx], upsamp_factor)) for idx in range(f1_coords.shape[0])]
    img2_coords = [cv_point(fcoord2imgcoord_center(f2_coords[idx], upsamp_factor)) for idx in range(f2_coords.shape[0])]
    cv_matches = [cv_match(i, i) for i in range(len(f1_coords))]
    match_result = cv2.drawMatches(img1, img1_coords, img2, img2_coords,
                                   cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_result


def fcoord2imgcoord_center(fcoord, samp=8):
    x = fcoord[0] * samp  # + samp // 2
    y = fcoord[1] * samp  # + samp // 2
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


def coord_in_map(loc_gps, map_geo, map_size):
    lon, lat = loc_gps
    left, top, right, bottom = map_geo
    w, h = map_size
    lon_per_px = (right - left) / w
    lat_per_px = (top - bottom) / h
    return (lon - left) / lon_per_px, (top - lat) / lat_per_px


def save_vis(id, frame1, frame2, savedir, coords=None, gt_corners=None, bbox_b=None):
    """
    INPUTS:
     - pred: predicted patch, a 3xpatch_sizexpatch_size tensor
     - gt2: GT patch, a 3xhxw tensor
     - gt1: first GT frame, a 3xhxw tensor
     - gt_grey: whether to use ground trught L channel in predicted image
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    b = frame1.size(0)
    frame1 = frame1 * 128 + 128
    frame2 = frame2 * 128 + 128

    for cnt in range(b):

        im = frame1[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_frame1 = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        im = frame2[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_frame2 = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        if bbox_b is not None:
            new_bbox = bbox_b[cnt]
            im_frame2 = draw_bbox(im_frame2, new_bbox)
            corners = gt_corners[cnt].cpu().detach().numpy()
            coord_img = coords[cnt].cpu().detach().numpy()
            frame1_f_size = frame1.size(-1) // 8
            frame1_f_grid_x = np.arange(0, frame1_f_size).reshape((1, -1))
            frame1_f_grid_y = frame1_f_grid_x.reshape((-1, 1))
            frame1_f_grid_x = np.repeat(frame1_f_grid_x, frame1_f_size, axis=0)
            frame1_f_grid_y = np.repeat(frame1_f_grid_y, frame1_f_size, axis=1)
            frame1_f_grid_x = np.expand_dims(frame1_f_grid_x, axis=-1)
            frame1_f_grid_y = np.expand_dims(frame1_f_grid_y, axis=-1)
            frame1_f_grid = np.concatenate([frame1_f_grid_x, frame1_f_grid_y], axis=-1)
            frame1_f_grid_flat = frame1_f_grid.reshape((-1, 2))
            dsamp_mask = np.zeros(frame1_f_grid_x.shape[:2]) != 0
            dsamp_mask[::8, ::8] = True
            dsamp_mask_flat = dsamp_mask.reshape((-1,))
            frame1_f_coords = frame1_f_grid_flat[dsamp_mask_flat]
            estimate_coords = coord_img[dsamp_mask_flat]

            cv2.polylines(im_frame2, np.expand_dims(corners.astype(np.int32), axis=0), 1, (0, 0, 255), 3)

            match_img = draw_matches(im_frame1, im_frame2, frame1_f_coords, estimate_coords, upsamp_factor=8)

            cv2.imwrite(os.path.join(savedir, str(id) + "_{:02d}_match.png".format(cnt)), match_img)


def line_batch(bpt1, bpt2):
    bk = (bpt2[:, 1] - bpt1[:, 1]) / (bpt2[:, 0] - bpt1[:, 0])
    return bk, bpt1[:, 1] - bk * bpt1[:, 0]


def cross_pt_batch(bline1, bline2):
    k1, b1 = bline1
    k2, b2 = bline2
    x = (b2 - b1) / (k1 - k2)
    return x.unsqueeze(dim=1), (x * k1 + b1).unsqueeze(dim=1)


def avg_px_dist_of_batch(bbox, corners_gt):
    bbox_lt = torch.cat([bbox[:, :1], bbox[:, 1:2]], dim=-1).unsqueeze(1)
    bbox_rb = torch.cat([bbox[:, 2:3], bbox[:, 3:]], dim=-1).unsqueeze(1)
    bbox_centers = torch.cat([bbox_lt, bbox_rb], dim=1).mean(1)
    bline1 = line_batch(corners_gt[:, 0, :], corners_gt[:, 2, :])
    bline2 = line_batch(corners_gt[:, 1, :], corners_gt[:, 3, :])
    x_b, y_b = cross_pt_batch(bline1, bline2)
    est_pts = torch.cat([x_b, y_b], dim=-1)
    square = (est_pts - bbox_centers) ** 2
    dist_t = (torch.sum(square, dim=-1) ** 0.5).mean()
    return dist_t.unsqueeze(dim=0).cpu().detach().numpy()[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def avg_iou_of_batch(bbox, corners_gt):
    # left,top,right,bottom
    batch_size = bbox.size(0)
    bbox_np = bbox.cpu().detach().numpy()
    corners_gt_np = corners_gt.cpu().detach().numpy()
    corners_est_np = np.concatenate([np.expand_dims(np.concatenate([bbox_np[:, :1], bbox_np[:, 1:2]], axis=-1), axis=1),
                                     np.expand_dims(np.concatenate([bbox_np[:, 2:3], bbox_np[:, 1:2]], axis=-1),
                                                    axis=1),
                                     np.expand_dims(np.concatenate([bbox_np[:, 2:3], bbox_np[:, 3:]], axis=-1), axis=1),
                                     np.expand_dims(np.concatenate([bbox_np[:, :1], bbox_np[:, 3:]], axis=-1),
                                                    axis=1), ],
                                    axis=1)
    iou_sum = 0.
    for i in range(batch_size):
        poly_box = Polygon(corners_est_np[i]).convex_hull
        poly_gt = Polygon(corners_gt_np[i])
        poly_union = MultiPoint(np.concatenate([corners_est_np[i], corners_gt_np[i]])).convex_hull
        if not poly_box.intersects(poly_gt):
            continue
        inter_area = poly_box.intersection(poly_gt).area
        union_area = poly_union.area
        iou_sum += float(inter_area) / union_area
    return iou_sum / batch_size


def invariant_test_match(args):
    model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp, Resnet=args.Resnet,
                  color_switch=False, coord_switch=False, model_scale=args.estimate_scale)
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.cuda()
    model.eval()

    aug_scale = [{"scale": (i / 10,)} for i in range(5, 15)]
    aug_rot = [{"rotate": (i * 10,)} for i in range(19)]
    for aug in aug_scale:
        scale_factor = aug['scale'][0]
        _, dataset, _ = vhr.getVHRRemoteDataAugCropper(aug=aug)
        dist_avg_meter = AverageMeter()
        iou_avg_meter = AverageMeter()
        dataloader = DataLoader(dataset, batch_size=args.batchsize)
        expr_dir = os.path.join('./experiments/scale_vhr', str(scale_factor))
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(expr_dir, 'test.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        for i, frames in enumerate(dataloader):
            frame1_var = frames[0].cuda()
            frame2_var = frames[1].cuda()
            corners_gt = frames[-1].cuda()
            p_size = args.patch_size // 8
            model.grid_flat = None
            esti_bbox, _, coords = model(frame1_var, frame2_var, warm_up=False, patch_size=(p_size, p_size),
                                         test_result=True)

            if frame1_var.size(2) > args.patch_size:
                center = frame1_var.size(2) // 2
                half_patch_size = args.patch_size // 2
                bbox = torch.tensor(
                    [center - half_patch_size, center - half_patch_size, center - half_patch_size + args.patch_size,
                     center - half_patch_size + args.patch_size], dtype=torch.float).cuda()
                bbox = bbox.repeat(args.batchsize, 1)
                frame1_var = diff_crop(frame1_var, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                       args.patch_size, args.patch_size)

            save_vis(i, frame1_var, frame2_var, expr_dir, coords, corners_gt, esti_bbox)
            dist = avg_px_dist_of_batch(esti_bbox, corners_gt)
            dist_avg_meter.update(dist, args.batchsize)
            iou = avg_iou_of_batch(esti_bbox, corners_gt)
            iou_avg_meter.update(iou, args.batchsize)
            if i % 8 == 0:
                logger.info(str(i) + '-dist-' + str(dist))
                logger.info(str(i) + '-iou-' + str(iou))
        logger.info('scale ' + str(scale_factor) + ' avg_dist: ' + str(dist_avg_meter.avg))
        logger.info('scale ' + str(scale_factor) + ' avg_iou: ' + str(iou_avg_meter.avg))
    for aug in aug_rot:
        rot_deg = aug['rotate'][0]
        _, dataset, _ = vhr.getVHRRemoteDataAugCropper(aug=aug)
        dist_avg_meter = AverageMeter()
        iou_avg_meter = AverageMeter()
        dataloader = DataLoader(dataset, batch_size=args.batchsize)
        expr_dir = os.path.join('./experiments/rot_vhr', str(rot_deg))
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(expr_dir, 'test.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        for i, frames in enumerate(dataloader):
            frame1_var = frames[0].cuda()
            frame2_var = frames[1].cuda()
            corners_gt = frames[-1].cuda()
            p_size = args.patch_size // 8
            model.grid_flat = None
            esti_bbox, _, coords = model(frame1_var, frame2_var, warm_up=False, patch_size=(p_size, p_size),
                                         test_result=True)

            if frame1_var.size(2) > args.patch_size:
                center = frame1_var.size(2) // 2
                half_patch_size = args.patch_size // 2
                bbox = torch.tensor(
                    [center - half_patch_size, center - half_patch_size, center - half_patch_size + args.patch_size,
                     center - half_patch_size + args.patch_size], dtype=torch.float).cuda()
                bbox = bbox.repeat(args.batchsize, 1)
                frame1_var = diff_crop(frame1_var, bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3],
                                       args.patch_size, args.patch_size)

            save_vis(i, frame1_var, frame2_var, expr_dir, coords, corners_gt, esti_bbox)
            dist = avg_px_dist_of_batch(esti_bbox, corners_gt)
            dist_avg_meter.update(dist, args.batchsize)
            iou = avg_iou_of_batch(esti_bbox, corners_gt)
            iou_avg_meter.update(iou, args.batchsize)
            if i % 8 == 0:
                logger.info(str(i) + '-dist-' + str(dist))
                logger.info(str(i) + '-iou-' + str(iou))
        logger.info('rotate ' + str(rot_deg) + ' avg_dist: ' + str(dist_avg_meter.avg))
        logger.info('rotate ' + str(rot_deg) + ' avg_iou: ' + str(iou_avg_meter.avg))


if __name__ == '__main__':
    args = parse_args()
    invariant_test_match(args)
