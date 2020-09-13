import cv2 as cv
import numpy as np

from data.sensefly import scene_list, SenseflyGeoDataReader


def default_corners(img):
    return np.array([(0, 0), (img.shape[1] - 1, 0),
                     (img.shape[1] - 1, img.shape[0] - 1), (0, img.shape[0] - 1)])


def mask_of(h, w, corners):
    mask = np.zeros((h, w), np.uint8)
    corners = corners.reshape((-1, 1, 2))
    cv.polylines(mask, [corners], True, 255)
    cv.fillPoly(mask, [corners], 255)
    return mask


def detect_compute(img, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    detector = cv.xfeatures2d_SURF.create(400)  # cv.xfeatures2d_SIFT.create()
    sift_points, sift_desc = detector.detectAndCompute(img, mask_of(img.shape[0], img.shape[1], content_corners))
    return sift_points, sift_desc


def feature_match(img1, img2, corners1=None, corners2=None, img1_features=None, img2_features=None, draw=False):
    ratio_thresh = 0.75
    if img1_features is None:
        img1_points, img1_desc = detect_compute(img1, corners1)
    else:
        img1_points, img1_desc = img1_features[0], img1_features[1]
    if img2_features is None:
        img2_points, img2_desc = detect_compute(img2, corners2)
    else:
        img2_points, img2_desc = img2_features[0], img2_features[1]
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    raw_matches = matcher.knnMatch(img1_desc, img2_desc, 2)
    good_matches = []
    # match filtering
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    corresponding1 = []
    corresponding2 = []
    for match in good_matches:
        corresponding1.append(img1_points[match.queryIdx].pt)
        corresponding2.append(img2_points[match.trainIdx].pt)
    if len(corresponding1) < 4:
        retval = None
        filtered_matches = []
    else:
        corresponding1 = np.float32(corresponding1).reshape(-1, 1, 2)
        corresponding2 = np.float32(corresponding2).reshape(-1, 1, 2)
        retval, mask = cv.findHomography(corresponding1, corresponding2, method=cv.RANSAC, ransacReprojThreshold=5,
                                         maxIters=4096)
        filtered_matches = []
        for i in range(mask.shape[0]):
            if mask[i][0] == 1:
                filtered_matches.append(good_matches[i])
    if draw:
        draw_img = draw_match(img1, img1_points, img2, img2_points, filtered_matches)
        return retval, draw_img
    return retval


def draw_match(img1, img1_points, img2, img2_points, matches, save_path=None):
    result = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, img1_points, img2, img2_points, matches, result,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if save_path is not None:
        cv.imwrite(save_path, result)
    else:
        return result


def coord_in_map(loc_gps, map_geo, map_size):
    lon, lat = loc_gps
    left, top, right, bottom = map_geo
    w, h = map_size
    lon_per_px = (right - left) / w
    lat_per_px = (top - bottom) / h
    return (lon - left) / lon_per_px, (top - lat) / lat_per_px


if __name__ == '__main__':
    import logging
    import os
    from common import loc_dist, loc_of_corresponding

    expr_dir = './experiments/on_stitched_surf_match'
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

    for scene in scene_list:
        logger.info(scene)
        data_reader = SenseflyGeoDataReader('../Datasets/SenseFlyGeo', [scene], uav_scale_f=0.25)
        map_arrs = None
        map_features = None
        map_geo = None
        map_size = None
        save_dir = os.path.join(expr_dir, 'visualization0_25', scene)
        error_sum = None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(len(data_reader)):
            (uav_img, uav_loc, uav_fname), map_geo_tuples = data_reader[i]
            if map_arrs is None:
                map_arrs = [cv.cvtColor(map_geo_tuples[i][0], cv.COLOR_RGB2BGR) for i in range(len(map_geo_tuples))]
                map_features = [detect_compute(map_arr) for map_arr in map_arrs]
                map_fnames = [map_tuple[2] for map_tuple in map_geo_tuples]
                map_geo = map_geo_tuples[0][1]
                map_size = (map_geo_tuples[0][0].shape[1], map_geo_tuples[0][0].shape[0])
                error_sum = np.zeros(len(map_geo_tuples))
            uav_img = cv.cvtColor(uav_img, cv.COLOR_RGB2BGR)
            img_h, img_w = uav_img.shape[:2]
            for j in range(len(map_arrs)):
                logger.info('img ' + uav_fname + ' map ' + map_fnames[j])
                uav_img_features = detect_compute(uav_img)
                save_match = os.path.join(save_dir, 'map_' + map_fnames[j] + '_img_' + uav_fname + '.jpg')
                hmg, match_img = feature_match(uav_img, map_arrs[j], img1_features=uav_img_features,
                                               img2_features=map_features[j],
                                               draw=True)
                if hmg is None:
                    logger.info('Failed !')
                    continue
                logger.info('Hmg:')
                logger.info(hmg)
                uav_img_center = ((img_w - 1) / 2, (img_h - 1) / 2)
                est_coord, est_gps_loc = loc_of_corresponding(uav_img_center, hmg, map_geo, map_size)
                gt_coord = coord_in_map(uav_loc, map_geo, map_size)

                match_img[int(uav_img_center[1] - 15):int(uav_img_center[1] + 15),
                int(uav_img_center[0] - 15):int(uav_img_center[0] + 15), :] = (0, 0, 255)
                match_img[int(est_coord[1] - 15):int(est_coord[1] + 15),
                int(uav_img.shape[1] + est_coord[0] - 15):int(uav_img.shape[1] + est_coord[0] + 15), :] = (0, 0, 255)
                match_img[int(gt_coord[1] - 15):int(gt_coord[1] + 15),
                int(uav_img.shape[1] + gt_coord[0] - 15):int(uav_img.shape[1] + gt_coord[0] + 15), :] = (0, 255, 0)

                cv.imwrite(save_match, match_img)

                logger.info(est_gps_loc)
                error_dist = loc_dist(uav_loc, est_gps_loc)
                error_sum[j] += error_dist
                logger.info('Error:')
                logger.info(error_dist)
        error_mean = error_sum / len(data_reader)
        logger.info('Mean error:')
        logger.info(error_mean)
