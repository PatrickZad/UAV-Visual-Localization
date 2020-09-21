from data.sensefly import *
import math


def loc_dist(loc1, loc2):
    r""" Calculate 2D euclidean distance of 2 points given their GPS coordinates.
    Args:
        loc1: (lon1, lat1), GPS coordinate of point_1.
        loc2: (lon2, lat2), GPS coordinate of point_2.
    Returns:
        dist: 2D euclidean distance of loc1 and loc2 in meters.
    """
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    mean_r = 6372797  # meters
    fi1 = lat1 * math.pi / 180
    fi2 = lat2 * math.pi / 180
    d_fi = (lat2 - lat1) * math.pi / 180
    d_lam = (lon2 - lon1) * math.pi / 180
    a = math.sin(d_fi / 2) * math.sin(d_fi / 2) + math.cos(fi1) * math.cos(fi2) * \
        math.sin(d_lam / 2) * math.sin(d_lam / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return mean_r * c


def loc_of_corresponding(pt1, hmg, map_geo, map_size):
    pt1_h = np.array((pt1[0], pt1[1], 1))
    pt2_h = np.matmul(pt1_h, hmg.T)
    pt2_h /= pt2_h[-1]
    pt2 = (pt2_h[0], pt2_h[1])
    map_w, map_h = map_size
    left, top, right, bottom = map_geo
    return pt2, (left + (right - left) / map_w * pt2[0], top - (top - bottom) / map_h * pt2[1])


def decimal_gps(lon_tuple, lat_tuple):
    lon = lon_tuple[0] + lon_tuple[1] / 60 + lon_tuple[2] / 3600
    lat = lat_tuple[0] + lat_tuple[1] / 60 + lat_tuple[2] / 3600
    return float(lon), float(lat)
