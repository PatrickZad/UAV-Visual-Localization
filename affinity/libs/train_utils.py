import os
import cv2
import torch
import shutil
# import visdom
import numpy as np
from affinity.libs.vis_utils import draw_certainty_map, flow_to_rgb, prepare_img
from os.path import join
from random import randint
from shapely.geometry import Polygon, MultiPoint


def draw_bbox(img, bbox):
    """
    INPUTS:
     - segmentation, h * w * 3 numpy array
     - bbox: left, top, right, bottom
    OUTPUT:
     - image with a drawn bbox
    """
    # print("bbox: ", bbox)
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    color = np.array([51, 255, 255], dtype=np.uint8)
    c = tuple(map(int, color))
    img = cv2.rectangle(img, pt1, pt2, c, 5)
    return img


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


def draw_matches(img1, img2, f1_coords, f2_coords, upsamp_factor=8):
    # cv2.imwrite('./img1.jpg', img1)
    # cv2.imwrite('./img2.jpg', img2)
    img1_coords = [cv_point(fcoord2imgcoord_center(f1_coords[idx], upsamp_factor)) for idx in range(f1_coords.shape[0])]
    img2_coords = [cv_point(fcoord2imgcoord_center(f2_coords[idx], upsamp_factor)) for idx in range(f2_coords.shape[0])]
    cv_matches = [cv_match(i, i) for i in range(len(f1_coords))]
    match_result = cv2.drawMatches(img1, img1_coords, img2, img2_coords,
                                   cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_result


def save_vis(id, pred2, gt2, frame1, frame2, savedir, coords=None, gt_corners=None, new_c=None):
    """
    INPUTS:
     - pred: predicted patch, a 3xpatch_sizexpatch_size tensor
     - gt2: GT patch, a 3xhxw tensor
     - gt1: first GT frame, a 3xhxw tensor
     - gt_grey: whether to use ground trught L channel in predicted image
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    b = pred2.size(0)
    pred2 = pred2 * 128 + 128
    gt2 = gt2 * 128 + 128
    frame1 = frame1 * 128 + 128
    frame2 = frame2 * 128 + 128

    for cnt in range(b):
        im = pred2[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_bgr = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)
        im_pred = np.clip(im_bgr, 0, 255)

        im = gt2[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_gt2 = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        im = frame1[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_frame1 = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        im = frame2[cnt].cpu().detach().numpy().transpose(1, 2, 0)
        im_frame2 = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        if new_c is not None:
            new_bbox = new_c[cnt]
            im_frame2 = draw_bbox(im_frame2, new_bbox)
            '''cat_img = np.zeros((im_frame2.shape[0], im_frame1.shape[1] + im_frame2.shape[1], im_frame2.shape[2]))
            cat_img = cat_img.astype(np.uint8)
            cat_img[:im_frame1.shape[0], :im_frame1.shape[1], :] = im_frame1
            cat_img[:im_frame2.shape[0], im_frame1.shape[1]:, :] = im_frame2'''
            # im_frame2 = cv2.resize(im_frame2, (im_frame1.shape[0], im_frame1.shape[1]))
            # im = np.concatenate((im_frame1, im_frame2), axis=1)
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
            dsamp_mask[::4, ::4] = True
            dsamp_mask_flat = dsamp_mask.reshape((-1,))
            frame1_f_coords = frame1_f_grid_flat[dsamp_mask_flat]
            estimate_coords = coord_img[dsamp_mask_flat]

            cv2.polylines(im_frame2, np.expand_dims(corners.astype(np.int32), axis=0), 1, (0, 0, 255), 3)

            match_img = draw_matches(im_frame1, im_frame2, frame1_f_coords, estimate_coords, upsamp_factor=8)

            cv2.imwrite(os.path.join(savedir, str(id) + "_{:02d}_loc.png".format(cnt)), match_img)

        im = np.concatenate((im_frame1, im_pred, im_gt2), axis=1)
        cv2.imwrite(os.path.join(savedir, str(id) + "_{:02d}_patch.png".format(cnt)), im)


def save_vis_ae(pred, gt, savepath):
    b = pred.size(0)
    for cnt in range(b):
        im = pred[cnt].cpu().detach() * 128 + 128
        im = im.numpy().transpose(1, 2, 0)
        im_pred = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)
        # im_pred = np.clip(im_bgr, 0, 255)

        im = gt[cnt].cpu().detach() * 128 + 128
        im = im.numpy().transpose(1, 2, 0)
        im_gt = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_LAB2BGR)

        im = np.concatenate((im_gt, im_pred), axis=1)
        cv2.imwrite(os.path.join(savepath, "{:02d}.png".format(cnt)), im)


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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", savedir="models"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(savedir, 'model_best.pth.tar'))


def sample_patch(b, h, w, patch_size):
    left = randint(0, max(w - patch_size, 1))
    top = randint(0, max(h - patch_size, 1))
    right = left + patch_size
    bottom = top + patch_size
    return torch.Tensor([left, right, top, bottom]).view(1, 4).repeat(b, 1).cuda()


def log_current(epoch, loss_ave, best_loss, filename="log_current.txt", savedir="models"):
    file = join(savedir, filename)
    with open(file, "a") as text_file:
        print("epoch: {}".format(epoch), file=text_file)
        print("best_loss: {}".format(best_loss), file=text_file)
        print("current_loss: {}".format(loss_ave), file=text_file)


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
