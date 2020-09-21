import data.sensefly as sf
import data.augmentation as aug
import cv2 as cv
from affinity.model import track_match_comb as Model
from affinity.libs.test_utils import *
import argparse
import torch.nn as nn
import os
from common import loc_dist
from math import ceil


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

    parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=1024, help="full size for one frame.")
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


def transform_space(img, scale_n=6, rot_n=18):
    # cv.imwrite('./org.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))

    img_space = []
    rot_degs = range(360 // rot_n, 360, 360 // rot_n)
    img_h, img_w = img.shape[:2]
    center = ((img_w - 1) / 2, (img_h - 1) / 2)
    if img_h < img_w:
        width = img_h // 8 * 8
        start = int(center[0] - width / 2)
        img_space.append((0, 0, img[:, start:start + img_h, :].copy()))
    else:
        width = img_w // 8 * 8
        start = int(center[1] - width / 2)
        img_space.append((0, 0, img[start:start + img_w, :, :].copy()))
    for i in range(1, scale_n):
        scaled_img = cv.resize(img, dsize=(0, 0), fx=1 / 2 ** (0.5 * i), fy=1 / 2 ** (0.5 * i))
        for deg in rot_degs:
            rot_img, corners = aug.adaptive_rot(scaled_img, rot=deg)
            '''
            h, w = rot_img.shape[:2]
            corners[:, 0][corners[:, 0] < 0] = 0
            corners[:, 1][corners[:, 1] < 0] = 0
            corners[:, 0][corners[:, 0] > (w - 1)] = w - 1
            corners[:, 1][corners[:, 1] > (h - 1)] = h - 1'''

            # cv.imwrite('./rot_scale.jpg', cv.cvtColor(rot_img, cv.COLOR_RGB2BGR))
            img_crop = center_square_crop(rot_img, corners, str(i) + '_' + str(deg))

            # cv.imwrite('./' + str(i) + '_' + str(deg) + 'crop.jpg', cv.cvtColor(img_crop, cv.COLOR_RGB2BGR))
            img_space.append((i, deg, img_crop))
    return img_space


def draw_pt(img, pts, color=(0, 0, 255)):
    for pt in pts:
        h, w = img.shape[:2]
        left = ceil(max(0, pt[0] - 5))
        right = int(min(w, pt[0] + 5))
        top = ceil(max(0, pt[1] - 5))
        bottom = int(min(h, pt[1] + 5))
        img[top:bottom, left:right, :] = color


def center_square_crop(img, corners, prefix=''):
    '''corners[:, 0][corners[:, 0] < 0] = 0
    corners[:, 1][corners[:, 1] < 0] = 0
    corners = np.int32(corners)'''
    left = corners[corners[:, 0] == corners[:, 0].min()].squeeze()
    right = corners[corners[:, 0] == corners[:, 0].max()].squeeze()
    top = corners[corners[:, 1] == corners[:, 1].min()].squeeze()
    bottom = corners[corners[:, 1] == corners[:, 1].max()].squeeze()

    # lr_line = line(left, right)
    # tb_line = line(top, bottom)
    center = (img.shape[1] / 2, img.shape[0] / 2)
    # center = cross_pt(lr_line, tb_line)
    center_positive = (1, center[1] - center[0])
    center_negative = (-1, center[1] + center[0])

    cp_cross_pts = valid_cross_pts(left, top, right, bottom, center_positive)
    cn_cross_pts = valid_cross_pts(left, top, right, bottom, center_negative)

    '''img_vis = img.copy()
    img_vis = cv.cvtColor(img_vis, cv.COLOR_RGB2BGR)
    draw_pt(img_vis, cp_cross_pts)
    draw_pt(img_vis, cn_cross_pts)
    draw_pt(img_vis, [left, top, right, bottom], (0, 255, 0))
    cv.imwrite('./' + prefix + 'cross.jpg', img_vis)'''

    cp_dist = dist(*cp_cross_pts)
    cn_dist = dist(*cn_cross_pts)

    cross_pts = cp_cross_pts if cp_dist < cn_dist else cn_cross_pts

    cross_pts_arr = np.array(cross_pts)
    width = int((cross_pts_arr[:, 0].max() - cross_pts_arr[:, 0].min() + 1) // 8 * 8)
    start = (int(center[0] - width / 2), int(center[1] - width / 2))
    return img[start[1]:start[1] + width, start[0]:start[0] + width, :].copy()


def dist(pt1, pt2):
    return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5


def valid_cross_pts(left, top, right, bottom, test_line):
    lt_line = line(left, top)
    rt_line = line(right, top)
    lb_line = line(left, bottom)
    rb_line = line(right, bottom)

    pts = []

    lt_cross_pt = cross_pt(lt_line, test_line)
    if lt_cross_pt[0] < left[0] or lt_cross_pt[0] > right[0] \
            or lt_cross_pt[1] > bottom[1] or lt_cross_pt[1] < top[1]:
        pts.append(cross_pt(lb_line, test_line))
        pts.append(cross_pt(rt_line, test_line))
    else:
        pts.append(lt_cross_pt)
        pts.append(cross_pt(rb_line, test_line))
    return pts


def cross_pt(line1, line2):
    k1, b1 = line1
    k2, b2 = line2
    if k1 == k2:
        return None
    x = (b2 - b1) / (k1 - k2)
    return x, x * k1 + b1


def line(pt1, pt2):
    k = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    return k, pt1[1] - k * pt1[0]


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


def save_vis(frame1, loc, frame2, geo, coords, bbox):
    frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
    frame2 = cv.cvtColor(frame2, cv.COLOR_RGB2BGR)

    bbox = bbox.squeeze().cpu().detach().numpy()
    frame2 = draw_bbox(frame2, bbox)
    coords = coords.squeeze().cpu().detach().numpy()
    frame1_f_size = frame1.shape[0] // 8
    frame1_f_grid_x = np.arange(0, frame1_f_size).reshape((1, -1))
    frame1_f_grid_y = frame1_f_grid_x.reshape((-1, 1))
    frame1_f_grid_x = np.repeat(frame1_f_grid_x, frame1_f_size, axis=0)
    frame1_f_grid_y = np.repeat(frame1_f_grid_y, frame1_f_size, axis=1)
    frame1_f_grid_x = np.expand_dims(frame1_f_grid_x, axis=-1)
    frame1_f_grid_y = np.expand_dims(frame1_f_grid_y, axis=-1)
    frame1_f_grid = np.concatenate([frame1_f_grid_x, frame1_f_grid_y], axis=-1)
    frame1_f_grid_flat = frame1_f_grid.reshape((-1, 2))
    dsamp_mask = np.zeros(frame1_f_grid_x.shape[:2]) != 0
    dsamp_mask[::4, ::4] = True
    dsamp_mask_flat = dsamp_mask.reshape((-1,))
    frame1_f_coords = frame1_f_grid_flat[dsamp_mask_flat]
    estimate_coords = coords[dsamp_mask_flat]

    est_center = np.mean(coords, axis=0) * 8
    map_h, map_w = frame2.shape[:2]
    left, top, right, bottom = geo
    est_loc = (left + (right - left) / map_w * est_center[0], top - (top - bottom) / map_h * est_center[1])
    gt_coord = coord_in_map(loc, geo, (map_w, map_h))
    est_center = np.int32(est_center)
    gt_coord = np.array(gt_coord, dtype=np.int32)
    frame2[gt_coord[1] - 5:gt_coord[1] + 5, gt_coord[0] - 5:gt_coord[0] + 5, :] = (0, 255, 0)
    frame2[est_center[1] - 5:est_center[1] + 5, est_center[0] - 5:est_center[0] + 5, :] = (0, 0, 255)
    match_img = draw_matches(frame1, frame2, frame1_f_coords, estimate_coords, upsamp_factor=8)
    err = loc_dist(est_loc, loc)
    return err, match_img


def transform_search_match(args):
    model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp, Resnet=args.Resnet,
                  color_switch=False, coord_switch=False, model_scale=args.estimate_scale)
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    model.cuda()
    model.eval()

    data_reader = sf.SenseflyGeoDataReader('../Datasets/SenseFlyGeo', sf.scene_list, uav_scale_f=0.25, map_scale_f=0.25)
    for (scene, uav_img_fname, map_fname), (uav_img_arr, uav_loc, map_img_arr, map_geo) in data_reader:

        # cv.imwrite('./uav.jpg', cv.cvtColor(uav_img_arr, cv.COLOR_RGB2BGR))
        # cv.imwrite('./map.jpg', cv.cvtColor(map_img_arr, cv.COLOR_RGB2BGR))

        expr_dir = os.path.join('./experiments/trans_search', scene, map_fname[:-4])
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        transformed_uav_imgs = transform_space(uav_img_arr)
        map_lab = cv.cvtColor(map_img_arr, cv.COLOR_RGB2LAB)
        map_t = sf.LAB_arr2norm_t(map_lab, 'cuda', True)
        match_dict = {}
        for s, d, uav_img in transformed_uav_imgs:
            img_lab = cv.cvtColor(uav_img, cv.COLOR_RGB2LAB)
            img_t = sf.LAB_arr2norm_t(img_lab, 'cuda', True)
            img_size = img_t.size(2)
            p_size = (img_size // 8) - 2
            model.grid_flat = None
            bbox, _, coords = model(img_t, map_t, warm_up=False, patch_size=(p_size, p_size), test_result=True)
            ref_uav_img = uav_img[8:-8, 8:-8, :].copy()
            err, match_img = save_vis(ref_uav_img, uav_loc, map_img_arr, map_geo, coords, bbox)
            match_dict[err] = (s, d, match_img)
        sort_errs = sorted(match_dict.keys())
        for i in range(5):
            s, d, match_img = match_dict[sort_errs[i]]
            err_str = str(sort_errs[i])
            float_idx = err_str.find('.')
            err_int = err_str[:float_idx]
            save_fname = uav_img_fname[:-4] + '_' + err_int + '_' + str(s) + '_' + str(d) + '.jpg'
            save_path = os.path.join(expr_dir, save_fname)
            cv.imwrite(save_path, match_img)


if __name__ == '__main__':
    args = parse_args()
    transform_search_match(args)
