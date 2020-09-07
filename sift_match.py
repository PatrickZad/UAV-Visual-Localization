import cv2 as cv
import numpy as np
from .dataset.sensefly import scene_list, SenseflyGeoDataReader


def default_corners(img):
    return np.array([(0, 0), (img.shape[1] - 1, 0),
                     (img.shape[1] - 1, img.shape[0] - 1), (0, img.shape[0] - 1)])


def mask_of(h, w, corners):
    mask = np.zeros((h, w), np.uint8)
    sequence_corners = np.array(
        [corners[0], corners[1], corners[3], corners[2]]).reshape((-1, 1, 2))
    cv.polylines(mask, [sequence_corners], True, 255)
    cv.fillPoly(mask, [sequence_corners], 255)
    return mask


def detect_compute(img, content_corners=None, draw=None):
    if content_corners is None:
        content_corners = default_corners(img)
    detector = cv.xfeatures2d_SIFT.create()
    sift_points, sift_desc = detector.detectAndCompute(img, mask_of(img.shape[0], img.shape[1], content_corners))
    if draw is not None:
        empty = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv.drawKeypoints(img, sift_points, empty)
        cv.imwrite(draw, empty)
    print("Detect complete !")
    return sift_points, sift_desc


def feature_match(img1, img2, corners1=None, corners2=None, img1_features=None, img2_features=None, draw=None,
                  match_result=None):
    ratio_thresh = 0.7
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

    if match_result is not None:
        match_result.append((img1_points, img2_points, good_matches))
    corresponding1 = []
    corresponding2 = []
    for match in good_matches:
        corresponding1.append(img1_points[match.queryIdx].pt)
        corresponding2.append(img2_points[match.trainIdx].pt)
    corresponding1 = np.float32(corresponding1).reshape(-1, 1, 2)
    corresponding2 = np.float32(corresponding2).reshape(-1, 1, 2)
    retval, mask = cv.findHomography(corresponding1, corresponding2, method=cv.RANSAC, ransacReprojThreshold=1,
                                     maxIters=4096)
    filtered_matches = []
    for i in range(mask.shape[0]):
        if mask[i][0] == 1:
            filtered_matches.append(good_matches[i])
    if draw is not None:
        draw_match(img1, img1_points, img2, img2_points, filtered_matches, draw)
    return retval


def draw_match(img1, img1_points, img2, img2_points, matches, save_path):
    result = np.empty(
        (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, img1_points, img2, img2_points, matches, result,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(save_path, result)


if __name__ == '__main__':
    for scene in scene_list:
        data_reader = SenseflyGeoDataReader('../Datasets/SenseFlyGeo', [scene])
        map_features=None
        for i in range(len(data_reader)):
            (uav_img,uav_loc),map_geo_pair=data_reader[i]
            for map_img,map_geo in map_geo_pair:
                pass

