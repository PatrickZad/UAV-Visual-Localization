import numpy as np
import cv2
import random
import math


def warp_pts(pts_array, homo_mat):
    homog_pts = np.concatenate([pts_array, np.ones((pts_array.shape[0], 1))], axis=-1)
    warp_homog_pts = np.matmul(homog_pts, homo_mat.T)
    warp_homog_pts /= warp_homog_pts[:, 2:]
    return warp_homog_pts[:, :-1]


def default_corners(img):
    return np.array([[0, 0], [img.shape[1] - 1, 0], [
        0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]])


def adaptive_rot(img_array, trans_pts=None, random=True, rot=None):
    corners = default_corners(img_array)
    if random:
        rot = np.random.random() * 360
    rot = np.deg2rad(rot)
    s, c = np.sin(rot), np.cos(rot)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rot_corners = warp_pts(corners, mat)
    x, y, w, h = cv2.boundingRect(np.int32(rot_corners))
    translation = np.array([[-x, -y]])
    corners = rot_corners + translation
    mat[0, -1], mat[1, -1] = -x, -y
    rot_img = cv2.warpPerspective(img_array, mat, (w, h))
    pts = warp_pts(trans_pts, mat)
    return rot_img, corners,pts


def center_square(img, content_corners):
    x_sort = np.int32(np.sort(content_corners[:, 0]))
    y_sort = np.int32(np.sort(content_corners[:, 1]))
    crop = img[y_sort[1]:y_sort[2] + 1, x_sort[1]:x_sort[2] + 1, :].copy()
    w, h = crop.shape[1], crop.shape[0]
    if w > h:
        diff = w - h
        offset = diff // 2
        crop = cv2.copyMakeBorder(crop, top=0, bottom=0, left=offset, right=diff - offset,
                                  borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    elif w < h:
        diff = h - w
        offset = diff // 2
        crop = cv2.copyMakeBorder(crop, top=offset, bottom=diff - offset, left=0, right=0,
                                  borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return crop


def rand_crop(img, crop_size):
    ih, iw = img.shape[:2]
    ch, cw = crop_size
    h_extent = ih - ch
    w_extent = iw - cw
    h_offset = np.random.randint(0, h_extent + 1)
    w_offset = np.random.randint(0, w_extent + 1)
    return img[h_offset:h_offset + ch, w_offset:w_offset + cw, :].copy()


def rand_erase(img, probability=0.5, sl=0.02, sh=0.1, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
    if random.uniform(0, 1) >= probability:
        return img
    ih, iw = img.shape[:2]
    for attempt in range(100):
        area = ih * iw

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < iw and h < ih:
            x1 = random.randint(0, ih - h)
            y1 = random.randint(0, iw - w)
            img[x1:x1 + h, y1:y1 + w, 0] = mean[0] * 128
            img[x1:x1 + h, y1:y1 + w, 1] = mean[1] * 128
            img[x1:x1 + h, y1:y1 + w, 2] = mean[2] * 128
        return img
    return img
