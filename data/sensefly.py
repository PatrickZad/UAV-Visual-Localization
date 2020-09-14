from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import re
import numpy as np
import cv2 as cv

scene_list = ['airport', 'gravel_pit', 'industrial1', 'industrial2',
              'new_road_corridor', 'technology_park', 'village']


class SenseflyGeoDataReader:
    r"""Class that reads UAV-captured images and region maps.
    All UAV-captured images are with GPS coordinates and other information in their metadata. All region maps were downloaded from Google Earth Pro
    and captured at different time. GPS coordinates of maps were recorded in map_geo.csv.
    """

    def __init__(self, dataset_dir, scene_list=scene_list, uav_crop_size=None, uav_scale_f=None, map_size=None,
                 map_scale_f=None, aug_methods=[]):
        assert uav_crop_size is None or uav_scale_f is None
        assert map_size is None or map_scale_f is None
        self._dir = dataset_dir
        self._scene_list = scene_list
        self._uav_crop_size = uav_crop_size
        self._uav_scale_f = uav_scale_f
        self._map_size = map_size
        self._map_scale_f = map_scale_f
        self._aug_methods = aug_methods
        self._scene_pairs = []

        for i in range(len(scene_list)):
            img_files = os.path.join(dataset_dir, self._scene_list[i], 'imgs')
            mapfiles=os.path.join(dataset_dir, self._scene_list[i], 'imgs')
            self._scene_pairs += [(img_file, i) for img_file in img_files]
        self._map_geo = pd.read_csv(os.path.join(dataset_dir, 'map_geo.csv'))

    def __len__(self) -> int:
        return len(self._scene_pairs)

    def _crop_uav_img(self, img):
        h, w = img.shape[:2]
        h_ratio = self._uav_crop_size[1] / h
        w_r = int(w * h_ratio)
        if w_r > self._uav_crop_size[0]:
            img_r = cv.resize(img, dsize=(w_r, self._uav_crop_size[1]))
            offset = int(img_r.shape[1] / 2 - (self._uav_crop_size[0] / 2))
            crop_img = img_r[:, offset:offset + self._uav_crop_size[0], :].copy()
        else:
            w_ratio = self._uav_crop_size[0] / w
            h_r = int(h * w_ratio)
            img_r = cv.resize(img, dsize=(h_r, self._uav_crop_size[0]))
            offset = int(img_r.shape[0] / 2 - (self._uav_crop_size[1] / 2))
            crop_img = img_r[offset:offset + self._uav_crop_size[1], :].copy()
        return crop_img

    def _pad_map_img(self, img, loc):
        h, w = img.shape[:2]
        left, top, right, bottom = loc
        h_ratio = self._map_size[1] / h
        w_r = int(w * h_ratio)
        if w_r < self._map_size[0]:
            img_r = cv.resize(img, dsize=(w_r, self._uav_crop_size[1]))
            left_padding = (self._map_size[0] - w_r) // 2
            right_padding = self._map_size[0] - w_r - left_padding
            left -= (right - left) / w_r * left_padding
            right += (right - left) / w_r * right_padding
            r_img = np.zeros((self._map_size[0], self._map_size[1], 3), dtype=np.uint8)
            r_img[:, left_padding:left_padding + w_r, :] = img_r
        else:
            w_ratio = self._map_size[0] / w
            h_r = int(h * w_ratio)
            img_r = cv.resize(img, dsize=(h_r, self._uav_crop_size[0]))
            top_padding = (self._map_size[1] - h_r) // 2
            bottom_padding = self._map_size[1] - h_r - top_padding
            top += (top - bottom) / h_r * top_padding
            bottom -= (top - bottom) / h_r * bottom_padding
            r_img = np.zeros((self._map_size[0], self._map_size[1], 3), dtype=np.uint8)
            r_img[top_padding:top_padding + h_r, :, :] = img_r
        return img_r, (left, top, right, bottom)

    def __getitem__(self, index: int):
        uav_img_fname, scene_id = self._scene_pairs[index]
        uav_pil_img = Image.open(os.path.join(self._dir, self._scene_list[scene_id], uav_img_fname))
        exifdata = uav_pil_img.getexif()
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            if re.match(r'.*GPS.*Latitude.*', tag):
                lat = exifdata.get(tag_id)
                continue
            if re.match(r'.*GPS.*Longitude.*', tag):
                lon = exifdata.get(tag_id)
                continue
        uav_img_arr = np.array(uav_pil_img)
        uav_loc = (lon, lat)
        if self._uav_crop_size is not None:
            uav_img_arr = self._crop_uav_img(uav_img_arr)
        elif self._uav_scale_f is not None:
            uav_img_arr = cv.resize(uav_img_arr, (0, 0), self._uav_scale_f, self._uav_scale_f)
        map_files = os.listdir(os.path.join(self._dir, self._scene_list[scene_id], 'maps'))
        map_img_arr_list = [
            cv.cvtColor(cv.imread(os.path.join(self._dir, self._scene_list[scene_id], 'maps', map_file)),
                        cv.COLOR_BGR2RGB) for map_file in map_files]
        map_img_geo = self._map_geo.loc[self._scene_list[scene_id], :4]
        if self._map_scale_f is not None:
            map_geo_pair = [(cv.resize(map_img_arr, (0, 0), self._map_scale_f, self._map_scale_f), map_img_geo) for
                            map_img_arr in map_img_arr_list]
        elif self._map_size is not None:
            map_geo_pair = [self._pad_map_img(map_img, map_img_geo) for map_img in map_img_arr_list]
        else:
            map_geo_pair = [(map_img, map_img_geo) for map_img in map_img_arr_list]
        return (uav_img_arr, uav_loc), map_geo_pair


class SenseFlyGeoDataset(Dataset):
    def __init__(self, data_reader: SenseflyGeoDataReader):
        self._data_reader = data_reader

    def __len__(self):
        return len(self._data_reader)

    def __getitem__(self, item):
        (uav_img, uav_loc), map_geo_pairs = self._data_reader[item]
        uav_img_t = torch.Tensor(np.transpose(uav_img, (2, 0, 1)))
        rand_map_id = np.random.randint(len(map_geo_pairs))
        map_img, map_geo = map_geo_pairs[rand_map_id]
        map_t = torch.Tensor(np.transpose(map_img, (2, 1, 0)))
        uav_img_t_norm = (uav_img_t - 128) / 128
        map_img_t_norm = (map_t - 128) / 128
        return uav_img_t_norm, uav_loc, map_img_t_norm, map_geo


class SenseFlyGeoSceneVal(Dataset):
    def __init__(self, data_reader: SenseflyGeoDataReader):
        pass

    def __getitem__(self, item):

        pass
    def __len__(self):
        pass
class SenseFlyGeoScenesReader(SenseflyGeoDataReader):
    def __init__(self,dataset_dir, scene_list=scene_list, uav_crop_size=None, uav_scale_f=None, map_size=None,
                 map_scale_f=None, aug_methods=[]):
        assert uav_crop_size is None or uav_scale_f is None
        assert map_size is None or map_scale_f is None
        self._dir = dataset_dir
        self._scene_list = scene_list
        self._uav_crop_size = uav_crop_size
        self._uav_scale_f = uav_scale_f
        self._map_size = map_size
        self._map_scale_f = map_scale_f
        self._aug_methods = aug_methods
        self._img_pairs = []