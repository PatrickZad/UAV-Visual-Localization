import numpy as np
import cv2
import random
import math
from skimage import transform


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
    if trans_pts is not None:
        pts = warp_pts(trans_pts, mat)
        return rot_img, corners, pts
    else:
        return rot_img, corners


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


def adaptive_affine(img, affine_mat, content_corners=None):
    # 2 by 2 mat
    # auto translation
    if content_corners is None:
        content_corners = default_corners(img)
    affined_corners = np.int32(np.matmul(content_corners, affine_mat.T))
    x, y, w, h = cv2.boundingRect(affined_corners)
    translation = np.array([-x, -y])
    for corner in affined_corners:
        corner += translation
    # return affined and translated corners,adaptive translation affine mat,bounding rectangular width and height
    affine_mat = np.concatenate([affine_mat, translation.reshape((2, 1))], axis=1)
    return affined_corners, affine_mat, (w, h)


def rotation_phi(img, phi, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if phi == 0:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    phi = np.deg2rad(phi)
    s, c = np.sin(phi), np.cos(phi)
    mat_rot = np.array([[c, -s], [s, c]])
    rot_corners, affine_mat, bounding = adaptive_affine(img, mat_rot, content_corners)
    affined = cv2.warpAffine(img, affine_mat, bounding)
    return affined, affine_mat, rot_corners


def tilt_image(img, tilt, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if tilt == 1:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    gaussian_sigma = 0.8 * np.sqrt(tilt ** 2 - 1)
    unti_aliasing = cv2.GaussianBlur(
        img, (3, 1), sigmaX=0, sigmaY=gaussian_sigma)
    mat_tilt = np.array([[1, 0], [0, 1 / tilt]])
    tilt_corners, affine_mat, bounding = adaptive_affine(img, mat_tilt, content_corners)
    affined = cv2.warpAffine(unti_aliasing, affine_mat, bounding)
    return affined, affine_mat, tilt_corners


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
