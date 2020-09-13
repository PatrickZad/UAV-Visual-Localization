import os
from os.path import dirname
import cv2
import torch
from torch.utils.data import Dataset
from common import *
from skimage import io, transform, draw
import cv2 as cv
import numpy as np
import data.augmentation as data_aug
import math

global iter
iter = 0


class ResiscDataset(Dataset):
    # 128*128 patches
    def __init__(self, file_path, device, categories):
        self.common_path = os.path.join(dataset_common_dir, 'NWPU-RESISC45')
        self.file_paths = file_path
        self.device = device
        self.categories = categories
        self.img_widths = range(224, 385)

    def __getitem__(self, idx):
        global iter
        img_array = io.imread(os.path.join(self.common_path, self.file_paths[idx][0],
                                           self.file_paths[idx][1]))  # h * w *c RGB image array
        img_class = self.file_paths[idx][0]
        class_idx = self.categories[img_class]
        '''# random rotation
        rot_img, corners = aug.adaptive_rot(img_array)
        # square patch
        crop = aug.center_square(rot_img, corners)'''
        # random scale
        idx = np.random.randint(0, len(self.img_widths))
        img_width = self.img_widths[idx]
        # random homography
        scaled_img = transform.resize(img_array, (img_width, img_width))
        x_offset, y_offset = 0, 0
        if img_width > 224:
            x_offset = np.random.randint(0, img_width - 224 + 1)
            y_offset = np.random.randint(0, img_width - 224 + 1)
        crop = scaled_img[y_offset:y_offset + 224,
               x_offset:x_offset + 224, :].copy()
        # io.imsave('./experiments/train_classifier/' + str(iter) + '.jpg', crop)

        data_array = np.transpose(crop, (2, 0, 1))
        # result = np.zeros(len(self.categories))
        # result[class_idx] = 1
        result = class_idx

        iter += 1

        return torch.tensor(data_array, dtype=torch.float, device=self.device), \
               torch.tensor(result, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.file_paths)


class OxfordBuildingsLocalization:
    # left-top x,y roght-bottpm x,y
    def __init__(self, test_num=2):
        self._img_dir = os.path.join(
            dataset_common_dir, 'OxfordBuilding', 'imgs')
        self._next = 0
        gt_dir = os.path.join(dataset_common_dir, 'OxfordBuilding', 'gt')
        self._query_bboxes = {}
        self._img_pairs = []
        for file in os.listdir(gt_dir):
            if 'query' in file:
                category = file[:-9]
                with open(os.path.join(gt_dir, file)) as f:
                    gt_str = f.readline().split('\n')[0]
                    info = gt_str.split(' ')
                    query_img = info[0][5:]
                    box_arr = np.array(info[1:], dtype=np.float)
                    query_box = np.int32(box_arr)
                    self._query_bboxes[query_img] = query_box
                with open(os.path.join(gt_dir, category + 'good.txt')) as f:
                    filenames = f.readlines()
                    if test_num < len(filenames):
                        idx = np.random.randint(
                            0, len(filenames), size=(test_num,))
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in idx]
                    else:
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in range(len(filenames))]
                with open(os.path.join(gt_dir, category + 'ok.txt')) as f:
                    filenames = f.readlines()
                    if test_num < len(filenames):
                        idx = np.random.randint(
                            0, len(filenames), size=(test_num,))
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in idx]
                    else:
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in range(len(filenames))]
                with open(os.path.join(gt_dir, category + 'junk.txt')) as f:
                    filenames = f.readlines()
                    if test_num < len(filenames):
                        idx = np.random.randint(
                            0, len(filenames), size=(test_num,))
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in idx]
                    else:
                        self._img_pairs += [(query_img, filenames[i][:-1])
                                            for i in range(len(filenames))]

    def __iter__(self):
        return self

    def __next__(self):
        if self._next == len(self._img_pairs):
            raise StopIteration
        query_img, test_img = self._img_pairs[self._next]
        query_img_arr = io.imread(os.path.join(
            self._img_dir, query_img + '.jpg'))
        test_img_arr = io.imread(os.path.join(
            self._img_dir, test_img + '.jpg'))
        bbox = self._query_bboxes[query_img]
        query_region_arr = query_img_arr[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1, :]
        self._next += 1
        return (query_img, test_img), (query_region_arr, test_img_arr)


def getResiscData(train_proportion=0.8, device='cpu'):
    data_base = os.path.join(dataset_common_dir, 'NWPU-RESISC45')
    scenes = os.listdir(data_base)
    categories = {scenes[i]: i for i in range(len(scenes))}
    indices = np.arange(0, 700)
    np.random.shuffle(indices)
    train_files = []
    val_files = []
    for scene in scenes:
        img_files = os.listdir(os.path.join(data_base, scene))
        for i in range(int(indices.shape[0] * train_proportion)):
            img_file = img_files[indices[i]]
            train_files.append((scene, img_file))
        for i in range(int(indices.shape[0] * train_proportion), indices.shape[0]):
            img_file = img_files[indices[i]]
            val_files.append((scene, img_file))
    return ResiscDataset(train_files, device, categories), ResiscDataset(val_files, device, categories)


class RemoteDataReader:
    def __init__(self):
        self.__ids = []
        with open(os.path.join(data_rs_dir, 'ids'), 'r') as id_file:
            for line in id_file.readlines():
                self.__ids.append(line[:-1])
        self.__length = len(self.__ids)
        self.__next_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__next_id == self.__length:
            raise StopIteration
        id = self.__ids[self.__next_id]
        map = io.imread(os.path.join(data_rs_dir, id + '.jpg'))
        query = io.imread(os.path.join(data_rs_dir, id + '_q.jpg'))
        self.__next_id += 1
        return id, map, query


aug_methods = ['scale', 'rotate', 'tilt', 'erase']
aug_light = {'scale': 1.5, 'rotate': 60}
aug_mid = {'scale': 2, 'rotate': 100, 'erase': (0.5, 0.01, 0.02, 0.6)}
aug_heavy = {'scale': 3, 'rotate': 120, 'erase': (0.7, 0.02, 0.05, 0.3), 'tilt': 0.1}


def rand(a=0, b=1, size=None):
    if size is not None:
        return np.random.rand(*size) * (b - a) + a
    else:
        return np.random.rand() * (b - a) + a


def sk_warpcrop(img, homo_mat, warpcrop_box):
    warpped_img = transform.warp(img, homo_mat)
    crop = warpped_img[warpcrop_box[1]:warpcrop_box[1] + warpcrop_box[3],
           warpcrop_box[0]:warpcrop_box[0] + warpcrop_box[2], :].copy()
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    if crop_h != warpcrop_box[3] or crop_w != warpcrop_box[2]:
        print('Regenerate random patch !')
        return None
    crop = crop * 255
    crop = crop.astype(np.uint8)
    return crop


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
        offset_x = np.random.randint(int(0.1 * w), int(w - crop_size - 0.1 * w))
        offset_y = np.random.randint(int(0.1 * h), int(h - crop_size - 0.1 * h))
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
            scale_factor = rand(1, aug_options['scale'])
            scaled_img = cv.resize(map_arr, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
            crop, crop_corners = self._rand_crop(scaled_img, crop_size)
            crop_corners = crop_corners / scale_factor

        else:
            crop, crop_corners = self._rand_crop(map_arr, crop_size)
        if 'rotate' in aug_options.keys():
            rot = rand(-1, 1) * aug_options['rotate']
            if rot < 0:
                rot += 360
            map_arr, corners, crop_corners = data_aug.adaptive_rot(map_arr, trans_pts=crop_corners, random=False,
                                                                   rot=rot)
            map_arr, factor = self._resize_keep_ratio(map_arr, map_size)
            crop_corners = crop_corners * factor
        if 'erase' in aug_options.keys():
            crop = data_aug.rand_erase(crop, *aug_options['erase'])
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
        return 'vhr', crop_t, 'crop' + str(crop_rgb.shape[0]), map_t, self._data_reader.img_name(item), (
            crop_rgb, map_arr_rgb)


class VHRRemoteWarm(VHRRemoteDataset):
    def __init__(self, data_reader: VHRRemoteDataReader, crop_size, map_size, pertube=None):
        super(VHRRemoteWarm, self).__init__(data_reader, crop_size, map_size, pertube=pertube)

    def __getitem__(self, item):
        ref_t, tar_t = super(VHRRemoteWarm, self).__getitem__(item, return_pair=True)
        return ref_t, tar_t


def getVHRRemoteDataRandomCropper(crop_size=288, map_size=1024, proportion=(0.8, 0.8, 0.2), aug=aug_methods):
    # warm,train,val
    dir = os.path.join(dataset_common_dir, 'VHR Remote Sensing')
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

    return VHRRemoteWarm(warm, crop_size, map_size), VHRRemoteDataset(train, crop_size, map_size), \
           VHRRemoteVal(val, crop_size, map_size)


def getVHRRemoteDataAugCropper(crop_size=288, map_size=1024, proportion=(0.8, 0.8, 0.2), aug=aug_light, pertube=64):
    # TODO rand val
    # warm,train,val
    dir = os.path.join(dataset_common_dir, 'VHR Remote Sensing')
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


class SenseflyTransTrain(Dataset):

    def __init__(self, hard_map_prob=0, crop_size=768, map_size=1024) -> None:

        self._dataset_dir = os.path.join(
            dataset_common_dir, 'sensefly_trans', 'train')
        self._dirnames = os.listdir(self._dataset_dir)
        pair_list = []
        for dirname in self._dirnames:
            imgs = os.listdir(os.path.join(self._dataset_dir, dirname, 'imgs'))
            for img in imgs:
                pair_list.append((dirname, img))
        self._pair_list = pair_list
        self._hard_map_prob = hard_map_prob
        self._crop_size = crop_size

    def __getitem__(self, index: int):
        pair = self._pair_list[index]
        rand = np.random.random()
        if rand < self._hard_map_prob:
            hard_maps = os.listdir(os.path.join(self._dataset_dir, pair[0], 'hard'))
            if len(hard_maps) > 1:
                map_arr = cv2.imread(
                    os.path.join(self._dataset_dir, pair[0], 'hard', hard_maps[np.random.randint(0, len(hard_maps))]))
            else:
                map_arr = cv2.imread(os.path.join(self._dataset_dir, pair[0], 'hard', hard_maps[0]))
        else:
            map_arr = cv2.imread(os.path.join(self._dataset_dir, pair[0], 'map.jpg'))
        img_arr = cv2.imread(os.path.join(self._dataset_dir, pair[0], 'imgs', pair[1]))
        scale_size = self._crop_size * (np.random.random() / 4 + 1)
        img_size = img_arr.shape[:2]
        target_size = (scale_size, scale_size / img_size[0] * img_size[1]) if img_size[0] < img_size[1] else (
            scale_size / img_size[1] * img_size[0], scale_size)
        scaled_img_arr = cv.resize(img_arr, dsize=target_size)
        crop = data_aug.rand_crop(scaled_img_arr, (self._crop_size, self._crop_size))
        erase = data_aug.rand_erase(crop)

        img_arr = cv2.cvtColor(erase, cv2.COLOR_BGR2LAB)
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(img_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)
        target_size = (self._map_size, self._map_size / map_arr.shape[1] * map_arr.shape[0]) if map_arr.shape[0] < \
                                                                                                map_arr.shape[1] else (
            self._map_size / map_arr.shape[0] * map_arr.shape[1], self._map_size)
        map_arr = cv2.resize(map_arr, dsize=(int(target_size[0]), int(target_size[1])))
        if target_size[0] > target_size[1]:
            short_l = map_arr.shape[0]
            new_l = (short_l // 8 + 1) * 8
            map_size = (target_size[0], new_l)
        else:
            short_l = map_arr.shape[1]
            new_l = (short_l // 8 + 1) * 8
            map_size = (new_l, target_size[1])
        background = np.zeros((map_size[1], map_size[0], 3))
        background = background.astype(np.uint8)
        diff = map_size[1] - map_arr.shape[0] if map_arr.shape[0] < map_arr.shape[1] else map_size[0] - \
                                                                                          map_arr.shape[1]
        offset = diff // 2
        if map_arr.shape[0] < map_arr.shape[1]:
            background[offset:offset + map_arr.shape[0], :map_arr.shape[1], :] = map_arr
        else:
            background[:map_arr.shape[0], offset:offset + map_arr.shape[1], :] = map_arr
        map_arr = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
        map_t = torch.from_numpy(map_arr.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(map_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)

        return img_arr, map_arr

    def __len__(self) -> int:
        return len(self._pair_list)


class SenseflyTransVal(Dataset):
    def __init__(self, scale_f=0.6) -> None:
        self._dataset_dir = os.path.join(
            dataset_common_dir, 'sensefly_trans', 'val')
        self._dirnames = os.listdir(self._dataset_dir)
        pair_list = []
        for dirname in self._dirnames:
            imgs = os.listdir(os.path.join(self._dataset_dir, dirname, 'imgs'))
            maps = os.listdir(os.path.join(self._dataset_dir, dirname, 'map'))
            for img in imgs:
                for map_file in maps:
                    pair_list.append((dirname, img, map_file))
        self._pair_list = pair_list
        self._scale_f = scale_f

    def __getitem__(self, index: int):
        pair = self._pair_list[index]
        img_arr = cv2.imread(os.path.join(self._dataset_dir, pair[0], 'imgs', pair[1]))
        img_arr = cv2.resize(img_arr, (0, 0), fx=self._scale_f, fy=self._scale_f)
        img_arr = img_arr[:, (img_arr.shape[1] - img_arr.shape[0]) // 2:(img_arr.shape[1] - img_arr.shape[0]) // 2 +
                                                                        img_arr.shape[0], :].copy() if img_arr.shape[
                                                                                                           0] < \
                                                                                                       img_arr.shape[
                                                                                                           1] else img_arr[
                                                                                                                   (
                                                                                                                           img_arr.shape[
                                                                                                                               0] -
                                                                                                                           img_arr.shape[
                                                                                                                               1]) // 2:(
                                                                                                                                                img_arr.shape[
                                                                                                                                                    0] -
                                                                                                                                                img_arr.shape[
                                                                                                                                                    1]) // 2 +
                                                                                                                                        img_arr.shape[
                                                                                                                                            1],
                                                                                                                   :,
                                                                                                                   :].copy()
        map_arr = cv2.imread(os.path.join(self._dataset_dir, pair[0], 'map', pair[2]))
        map_arr = cv2.resize(map_arr, (0, 0), fx=self._scale_f, fy=self._scale_f)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(img_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)

        map_arr = cv2.cvtColor(map_arr, cv2.COLOR_BGR2LAB)
        map_t = torch.from_numpy(map_arr.transpose(2, 0, 1).copy()).contiguous().float()
        for t, m, s in zip(map_t, [128, 128, 128], [128, 128, 128]):
            t.sub_(m).div_(s)

        return pair[0], img_t, pair[1], map_t, pair[2]

    def __len__(self) -> int:
        return len(self._pair_list)

    def get_dataset_dir(self):
        return self._dataset_dir


if __name__ == '__main__':
    train_data, test_data = getVHRRemoteDataRandomCropper()
    counter = 0
    for img in train_data:
        io.imsave(os.path.join(expr_base, 'data_reader',
                               str(counter) + '.jpg'), img)
        counter += 1
        print(counter)
