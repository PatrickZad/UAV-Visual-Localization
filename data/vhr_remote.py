import os
import cv2
import torch
from torch.utils.data import Dataset
from skimage import io, transform, draw
import cv2 as cv
import numpy as np
from data.augmentation import *
import math

aug_methods = ['scale', 'rotate', 'tilt', 'erase']
aug_light = {'scale': (1, 1.5), 'rotate': (0, 60)}
aug_mid = {'scale': (1, 2), 'rotate': (0, 100), 'erase': (0.5, 0.01, 0.02, 0.6)}
aug_heavy = {'scale': (1, 3), 'rotate': (0, 120), 'erase': (0.7, 0.02, 0.05, 0.3), 'tilt': 0.1}


class VHRRemoteDataReader:
    def __init__(self, dir, files, aug_options):
        self._dir = dir
        self._files = files
        self._lenth = len(files)
        self._next = 0
        self._aug = aug_options

    def __iter__(self):
        return self

    def _expand_w(self, top, bottom, left, right):
        y1, y2 = (left[1], right[1]) if left[1] < right[1] else (
            right[1], left[1])
        xl1 = math.ceil((y1 - left[1]) / (top[1] - left[1])
                        * (top[0] - left[0]) + left[0])
        xl2 = math.ceil(
            (y2 - left[1]) / (bottom[1] - left[1]) * (bottom[0] - left[0]) + left[0])
        xr1 = int((y1 - right[1]) / (top[1] - right[1])
                  * (top[0] - right[0]) + right[0])
        xr2 = int((y2 - right[1]) / (bottom[1] - right[1])
                  * (bottom[0] - right[0]) + right[0])
        if xr1 < 0:
            xr1 = xr2 + 1
        elif xr2 < 0:
            xr2 = xr1 + 1
        return max(xl1, xl2), min(xr1, xr2)

    def _expand_h(self, top, bottom, left, right):
        x1, x2 = (top[0], bottom[0]) if top[0] < bottom[0] else (
            bottom[0], top[0])
        yl1 = math.ceil((x1 - top[0]) / (left[0] - top[0])
                        * (left[1] - top[1]) + top[1])
        yr1 = math.ceil((x2 - top[0]) / (right[0] - top[0])
                        * (right[1] - top[1]) + top[1])
        yl2 = int((x1 - bottom[0]) / (left[0] - bottom[0])
                  * (left[1] - bottom[1]) + bottom[1])
        yr2 = int((x2 - bottom[0]) / (right[0] - bottom[0])
                  * (right[1] - bottom[1]) + bottom[1])
        if yl2 < 0:
            yl2 = yr2 + 1
        elif yr2 < 0:
            yr2 = yl2 + 1
        return max(yl1, yr1), min(yl2, yr2)

    def _is_side_hori_or_verti(self, corners):
        for i in range(3):
            if abs(corners[i][0] - corners[i + 1][0]) < 2 or abs(corners[i][1] - corners[i + 1][1]) < 2:
                return True
        return False

    def _rand_aug(self, aug_options, img, corners):
        img_arr = img
        content_corners = corners if corners is not None else default_corners(
            img)
        affine_mat = np.eye(3)
        if aug_methods[1] in aug_options:
            phi = 360 - 360 * np.random.rand()
            img_arr, t_mat, content_corners = rotation_phi(
                img_arr, phi, content_corners)
            t_mat = np.concatenate([t_mat, np.array([[0, 0, 1]])], axis=0)
            affine_mat = np.matmul(t_mat, affine_mat)
        if aug_methods[2] in aug_options:
            max_deg_cos = np.cos(60 * np.pi / 180)
            tilt = np.random.rand() * (1 / max_deg_cos - 1) + 1
            img_arr, t_mat, content_corners = tilt_image(
                img_arr, tilt, content_corners)
            t_mat = np.concatenate([t_mat, np.array([[0, 0, 1]])], axis=0)
            affine_mat = np.matmul(t_mat, affine_mat)
        return img_arr, affine_mat, content_corners

    def __len__(self):
        return self._lenth

    def img_name(self, idx):
        return self._files[idx]

    def _rand_crop(self, map_arr, crop_size=256):
        h, w = map_arr.shape[:2]
        offset_x = np.random.randint(int(0.1 * w), int(0.9 * w - crop_size))
        offset_y = np.random.randint(int(0.1 * h), int(0.9 * h - crop_size))
        crop = map_arr[offset_y:offset_y + crop_size, offset_x:offset_x + crop_size, :].copy()
        corners = np.array([(offset_x, offset_y), (offset_x + crop_size - 1, offset_y),
                            (offset_x + crop_size - 1, offset_y + crop_size - 1), (offset_x, offset_y + crop_size - 1)])
        return crop, corners

    def _read_rgb(self, idx):
        origin_img = cv.imread(os.path.join(
            self._dir, self._files[idx]))
        rgb_img = cv.cvtColor(origin_img, cv.COLOR_BGR2RGB)
        return rgb_img

    def _resize_keep_ratio(self, origin_img, long_size):
        oh, ow = origin_img.shape[:2]
        img_size_r = (long_size, long_size / ow * oh) if ow > oh else (long_size / oh * ow, long_size)
        resized_arr = cv.resize(origin_img, (int(img_size_r[0]), int(img_size_r[1])))
        factor = int(img_size_r[0]) / ow
        return resized_arr, factor

    def _square_padding(self, img, tran_pts):
        square_size = max(img.shape[0], img.shape[1])
        background = np.zeros((square_size, square_size, 3))
        background = background.astype(np.uint8)
        diff = square_size - img.shape[0] if img.shape[0] < img.shape[1] \
            else square_size - img.shape[1]
        offset = diff // 2
        if img.shape[0] < img.shape[1]:
            background[offset:offset + img.shape[0], :img.shape[1], :] = img
            offset_coord = np.array([(0, offset)])
        else:
            background[:img.shape[0], offset:offset + img.shape[1], :] = img
            offset_coord = np.array([(offset, 0)])
        return background, tran_pts + offset_coord

    def read_item(self, idx, map_size=1024, crop_size=256, aug_options=None):
        if aug_options is not None:
            return self._aug_pair(idx, aug_options, map_size, crop_size)
        origin_img = self._read_rgb(idx)
        map_arr = self._resize_keep_ratio(origin_img, map_size)
        background = self._square_padding(map_arr)
        crop = self._rand_crop(map_arr, crop_size)
        return crop, background

    def crop_pair(self, idx, map_size=1024, crop_size=256, pertube=None):
        if pertube is not None:
            return self._rand_aug_crop_pair(idx, map_size, crop_size, pertube)
        origin_img = self._read_rgb(idx)
        map_arr = self._resize_keep_ratio(origin_img, map_size)
        h, w = map_arr.shape[:2]
        offset_x = np.random.randint(int(0.1 * w), int(w - crop_size - 0.1 * w))
        offset_y = np.random.randint(int(0.1 * h), int(h - crop_size - 0.1 * h))
        crop1 = map_arr[offset_y:offset_y + crop_size, offset_x:offset_x + crop_size, :].copy()
        offset_x2, offset_y2 = map_arr.shape[1], map_arr.shape[0]
        while offset_x + offset_x2 + crop_size > map_arr.shape[1] or offset_y + offset_y2 + crop_size > map_arr.shape[
            0]:
            offset_x2 = np.random.randint(max(int(-0.3 * crop_size), -offset_x), int(0.3 * crop_size))
            offset_y2 = np.random.randint(max(int(-0.3 * crop_size), -offset_y), int(0.3 * crop_size))
        crop2 = map_arr[offset_y + offset_y2:offset_y + offset_y2 + crop_size,
                offset_x + offset_x2:offset_x + offset_x2 + crop_size, :].copy()
        return crop1, crop2

    def _rand_aug_crop_pair(self, idx, map_size=1024, crop_size=256, pertube=32):
        origin_img = self._read_rgb(idx)
        map_arr, _ = self._resize_keep_ratio(origin_img, map_size)
        h, w = map_arr.shape[:2]
        offset_x = np.random.randint(int(0.1 * w), int(w - crop_size - 0.1 * w))
        offset_y = np.random.randint(int(0.1 * h), int(h - crop_size - 0.1 * h))
        crop1 = map_arr[offset_y:offset_y + crop_size, offset_x:offset_x + crop_size, :].copy()
        offset_x2, offset_y2 = map_arr.shape[1], map_arr.shape[0]
        while offset_x + offset_x2 + crop_size > map_arr.shape[1] or \
                offset_y + offset_y2 + crop_size > map_arr.shape[0]:
            offset_x2 = np.random.randint(max(int(-0.3 * crop_size), -offset_x), int(0.3 * crop_size))
            offset_y2 = np.random.randint(max(int(-0.3 * crop_size), -offset_y), int(0.3 * crop_size))
        offset_box = (offset_x + offset_x2, offset_y + offset_y2, offset_x + offset_x2 + crop_size - 1,
                      offset_y + offset_y2 + crop_size - 1)  # left,top,right,bottom
        h_ab_4p = np.array([(rand(max(-offset_box[0], -pertube), min(pertube, map_arr.shape[1] - offset_box[0])),
                             rand(max(-offset_box[1], -pertube), min(pertube, map_arr.shape[0] - offset_box[1]))),
                            (rand(max(-offset_box[2], -pertube), min(pertube, map_arr.shape[1] - offset_box[2])),
                             rand(max(-offset_box[1], -pertube), min(pertube, map_arr.shape[0] - offset_box[1]))),
                            (rand(max(-offset_box[2], -pertube), min(pertube, map_arr.shape[1] - offset_box[2])),
                             rand(max(-offset_box[3], -pertube), min(pertube, map_arr.shape[0] - offset_box[3]))),
                            (rand(max(-offset_box[0], -pertube), min(pertube, map_arr.shape[1] - offset_box[0])),
                             rand(max(-offset_box[3], -pertube), min(pertube, map_arr.shape[0] - offset_box[3]))),
                            ])
        corners_1 = np.array([(offset_box[0], offset_box[1]), (offset_box[2], offset_box[1]),
                              (offset_box[2], offset_box[3]), (offset_box[0], offset_box[3])])
        corners_2 = corners_1 + h_ab_4p
        h_ab = (transform.estimate_transform('projective', corners_1, corners_2)).params
        h_ab = h_ab / h_ab[2][2]
        h_ba = np.linalg.inv(h_ab)
        # patch_1 = map_arr[offset_box[1]:offset_box[3] + 1, offset_box[0]:offset_box[2] + 1, :].copy()
        crop2 = sk_warpcrop(map_arr, h_ba, (offset_box[0], offset_box[1], crop_size, crop_size))
        return crop1, crop2

    def _test_corners_gt(self, img, corners, crop):
        draw_img = img.copy()
        cv2.polylines(draw_img, np.expand_dims(corners.astype(np.int32), axis=0), 1, (0, 0, 255), 3)
        result = np.zeros((img.shape[0], img.shape[1] + crop.shape[1], 3))
        result = result.astype(np.uint8)
        result[:crop.shape[0], :crop.shape[1], :] = crop
        result[:draw_img.shape[0], crop.shape[1]:, :] = draw_img
        return result

    def _aug_pair(self, idx, aug_options, map_size=1024, crop_size=256):
        # TODO random tilt
        origin_img = self._read_rgb(idx)
        map_arr, _ = self._resize_keep_ratio(origin_img, map_size)
        if 'scale' in aug_options.keys():
            if len(aug_options['scale']) == 1:
                scale_factor = aug_options['scale'][0]
            else:
                scale_factor = rand(aug_options['scale'][0], aug_options['scale'][1])
            map_h, map_w = map_arr.shape[:2]
            if scale_factor * map_h < 1.32 * crop_size or scale_factor * map_w < 1.32 * crop_size:
                scale_factor = crop_size * 1.32 / min(map_h, map_w)
            scaled_img = cv.resize(map_arr, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
            crop, crop_corners = self._rand_crop(scaled_img, crop_size)
            crop_corners = crop_corners / scale_factor

        else:
            crop, crop_corners = self._rand_crop(map_arr, crop_size)
        if 'rotate' in aug_options.keys():
            if len(aug_options['rotate']) == 1:
                rot = aug_options['rotate'][0]
            else:
                rot = rand(aug_options['rotate'][0], aug_options['rotate'][1])
            if rot < 0:
                rot += 360
            map_arr, corners, crop_corners = adaptive_rot(map_arr, trans_pts=crop_corners, random=False,
                                                          rot=rot)
            map_arr, factor = self._resize_keep_ratio(map_arr, map_size)
            crop_corners = crop_corners * factor
        if 'erase' in aug_options.keys():
            crop = rand_erase(crop, *aug_options['erase'])
        map_arr, crop_corners = self._square_padding(map_arr, crop_corners)
        return crop, map_arr, crop_corners

    def __next__(self, idx, aug_options):
        origin_img = cv.imread(os.path.join(
            self._dir, self._files[self._next]))
        origin_img = cv.cvtColor(origin_img, cv.COLOR_BGR2RGB)
        img_arr = origin_img.copy()
        h, w = origin_img.shape[:2]
        ratio = h / w if h < w else w / h
        origin_corners = default_corners(origin_img)
        affine_mat = np.eye(3)
        box_h, box_w = h, w
        corners = origin_corners.copy()
        while True:
            while self._is_side_hori_or_verti(corners):
                img_arr, trans_mat, corners = self._rand_aug(
                    self._aug, origin_img, origin_corners)
                affine_mat = np.matmul(trans_mat, affine_mat)
            left = corners[corners[:, 0] == corners[:, 0].min()][0]
            right = corners[corners[:, 0] == corners[:, 0].max()][0]
            top = corners[corners[:, 1] == corners[:, 1].min()][0]
            bottom = corners[corners[:, 1] == corners[:, 1].max()][0]
            # top,bottom,left,right
            box = [min(left[1], right[1]), max(left[1], right[1]),
                   min(top[0], bottom[0]), max(top[0], bottom[0])]
            box_h, box_w = box[1] - box[0], box[3] - box[2]
            box_copy = box.copy()
            if box_w < box_h:
                box[2], box[3] = self._expand_w(top, bottom, left, right)
            else:
                box[0], box[1] = self._expand_h(top, bottom, left, right)
            box_h, box_w = box[1] - box[0], box[3] - box[2]
            if min(box_h, box_w) < min(img_arr.shape[0], img_arr.shape[1]) / 8:
                corners = origin_corners
                affine_mat = np.eye(3)
                continue
            else:
                break
        len_factor_short = np.random.rand() * (1 / 6 - 1 / 7) + 1 / 9
        len_factor_long = np.random.rand() * (1 / 9 - 1 / 10) + 1 / 14
        len1 = min(img_arr.shape[:2]) * len_factor_short
        len2 = max(img_arr.shape[:2]) * len_factor_long
        len_h = int(min(min(len1, len2) if box_h <
                                           box_w else max(len1, len2), box_h)) - 1
        len_w = int(min(len1 if len_h == len2 else len2, box_w)) - 1
        offset_x = np.random.randint(0, box_w - len_w)
        offset_y = np.random.randint(0, box_h - len_h)
        crop = img_arr[box[0] + offset_y:box[0] + offset_y + len_h,
               box[2] + offset_x:box[2] + offset_x + len_w, :].copy()
        if aug_methods[0] in self._aug:
            # 4~8
            factor = 3 * np.random.rand() + 3
            crop = cv.resize(crop, dsize=(0, 0), fx=factor, fy=factor)
        crop_corners = np.array([[box[2] + offset_x, box[0] + offset_y],
                                 [box[2] + offset_x + len_w, box[0] + offset_y],
                                 [box[2] + offset_x + len_w,
                                  box[0] + offset_y + len_h],
                                 [box[2] + offset_x, box[0] + offset_y + len_h]])
        inv_mat = np.linalg.inv(affine_mat)
        inv_corners = warp_pts(crop_corners, inv_mat)
        x, y, w, h = cv.boundingRect(np.int32(inv_corners))
        self._next += 1
        '''rgb_img = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB)
        rr, cc = draw.polygon_perimeter([box[0], box[0], box[1], box[1]], [box[2], box[3], box[3], box[2]],
                                        shape=rgb_img.shape, clip=True)
        draw.set_color(rgb_img, [rr, cc], (255, 0, 0))
        rr, cc = draw.polygon_perimeter([box_copy[0], box_copy[0], box_copy[1], box_copy[1]],
                                        [box_copy[2], box_copy[3], box_copy[3], box_copy[2]],
                                        shape=rgb_img.shape, clip=True)
        draw.set_color(rgb_img, [rr, cc], (0, 255, 0))
        rr, cc = draw.polygon_perimeter(
            [box[0] + offset_y, box[0] + offset_y, box[0] + offset_y + len_h, box[0] + offset_y + len_h],
            [box[2] + offset_x, box[2] + offset_x + len_w, box[2] + offset_x + len_w, box[2] + offset_x],
            shape=rgb_img.shape, clip=True)
        draw.set_color(rgb_img, [rr, cc], (0, 0, 255))'''

        return origin_img, crop, (x, y, w, h)


class VHRRemoteDataset(Dataset):
    def __init__(self, data_reader: VHRRemoteDataReader, crop_size, map_size, aug_options=None, pertube=None):
        self._data_reader = data_reader
        self._crop_size = crop_size
        self._map_size = map_size
        self._aug_options = aug_options
        self._pertube = pertube

    def __len__(self):
        return self._data_reader.__len__()

    def __getitem__(self, item, return_rgb=False, return_pair=False):
        if return_pair:
            ref, tar = self._data_reader.crop_pair(item, self._map_size, self._crop_size, self._pertube)
        else:
            ref, tar, crop_corners_gt = self._data_reader.read_item(item, self._map_size, self._crop_size,
                                                                    self._aug_options)
        ref = cv.cvtColor(ref, cv.COLOR_RGB2LAB)
        tar = cv.cvtColor(tar, cv.COLOR_RGB2LAB)
        tar_t = torch.from_numpy(tar.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(tar_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)
        ref_t = torch.from_numpy(ref.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(ref_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)
        return_data = [ref_t, tar_t]
        if return_rgb:
            return_data += [ref, tar]
        if not return_pair:
            return_data += [crop_corners_gt]
        return return_data


class VHRRemoteVal(VHRRemoteDataset):
    def __init__(self, data_reader: VHRRemoteDataReader, crop_size, map_size):
        super(VHRRemoteVal, self).__init__(data_reader, crop_size, map_size)

    def __getitem__(self, item):
        '''crop_rgb, map_arr_rgb = self._data_reader.read_item(item)
        crop = cv.cvtColor(crop_rgb, cv.COLOR_RGB2LAB)
        map_arr = cv.cvtColor(map_arr_rgb, cv.COLOR_RGB2LAB)
        map_t = torch.from_numpy(map_arr.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(map_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)
        crop_t = torch.from_numpy(crop.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(crop_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)'''
        crop_t, map_t, crop_rgb, map_arr_rgb = super(VHRRemoteVal, self).__getitem__(item, True)
        return crop_t, 'crop' + str(crop_rgb.shape[0]), map_t, self._data_reader.img_name(item), (
            crop_rgb, map_arr_rgb)


class VHRRemoteWarm(VHRRemoteDataset):
    def __init__(self, data_reader: VHRRemoteDataReader, crop_size, map_size, pertube=None):
        super(VHRRemoteWarm, self).__init__(data_reader, crop_size, map_size, pertube=pertube)

    def __getitem__(self, item):
        ref_t, tar_t = super(VHRRemoteWarm, self).__getitem__(item, return_pair=True)
        return ref_t, tar_t


def getVHRRemoteDataAugCropper(dir='../Datasets/VHR Remote Sensing', crop_size=400, map_size=1424,
                               proportion=(0.3, 0.3, 0.3), aug=aug_light, pertube=64):
    # TODO rand val
    # warm,train,val
    dir_files = os.listdir(dir)
    length = len(dir_files)
    len1 = int(length * proportion[0])
    len2 = int(length * proportion[1])
    len3 = int(length * proportion[2])
    np.random.shuffle(dir_files)
    part1 = dir_files[:len1]
    np.random.shuffle(dir_files)
    part2 = dir_files[:len2]
    np.random.shuffle(dir_files)
    part3 = dir_files[:len3]
    warm = VHRRemoteDataReader(dir, part1, aug)
    train = VHRRemoteDataReader(dir, part2, aug)
    val = VHRRemoteDataReader(dir, part3, aug)

    return VHRRemoteWarm(warm, crop_size, map_size, pertube), VHRRemoteDataset(train, crop_size, map_size, aug), \
           VHRRemoteVal(val, crop_size, map_size)
