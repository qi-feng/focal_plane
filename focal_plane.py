#! /usr/bin/python
# from astropy.table import Table
import argparse
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sewpy
from matplotlib.patches import Ellipse

font = {'size': 14}
import matplotlib
# from sklearn import cluster
from sklearn.cluster import KMeans

matplotlib.rc('font', **font)

try:
    import cv2

    has_cv2 = True
except:
    print("Can't import cv2!!")
    has_cv2 = False

import sys

if not sys.version_info.major == 3:
    print("You are running python 2...")
    input = raw_input

# Original GR cam lens with 8mm focal length used
# pix2mm = 0.48244
# x_corners = np.array([1210,1220,1957,1945])
# y_corners = np.array([1374, 635, 648, 1385])
# center=np.array([np.mean(x_corners), np.mean(y_corners)])
# center at ~60 deg is 1583. , 1010.5


# New GR cam lens with 16mm focal length used (2x zoom in)
# pix2mm = 0.2449
#PIX2MM = 0.241
PIX2MM = 0.482

# these new corners mark the central module
# x_corners = np.array([1762,1761,1980,1982])
# y_corners = np.array([1175,954,952,1174])
# center=np.array([np.mean(x_corners), np.mean(y_corners)])
# center=np.array([1871.25, 1063.75])
# center at ~-5 deg is 1871.25, 1063.75

# x_corners = np.array([1782,1781,2000,2002])
# y_corners = np.array([1175,954,952,1174])
# center=np.array([np.mean(x_corners), np.mean(y_corners)])
# center=np.array([1891.25, 1063.75])
# center at ~60 deg is 1891.25, 1063.75
# center at ~75 deg is 1896.25, 1063.75

# Ref positions taken on 2021 Aug 22 at 60 deg Elevation
# Ref Central Module (CM) positions: 
CM_REF_60 = np.array([[1460, 856],
                   [1684.1, 856],
                   [1460, 1080.1],
                   [1684.1, 1080.1]], dtype='float32')

# Ref positions taken on 2022 June 3 at -5 deg Elevation
CM_REF_minus5 = np.array([[1525, 957],
 [1637.033, 957],
 [1525, 1069.033],
 [1637.033, 1069.033]], dtype='float32')

CM_REF = CM_REF_minus5

# Ref LED positions @60 deg EL:
# 2021 June
LED_REF_before2021Sep19 = np.array([[1092.8107,1470.39],
           [1094.8242, 485.8442],
           [2074.988, 1468.9617],
           [2077.7751, 488.2695]], dtype='float32')
#2021-09-19
LED_REF_2021Sep19 = np.array([[1085.0588,	1469.3252],
                    [1094.7145,	485.3215],
                    [2067.249 ,  1475.3965],
                    [2076.2295,	495.7331]], dtype='float32')

#2022-04-14 @-5 deg EL
LED_REF_2022Apr14 = np.array([[1085.7123,	1460.7432],
                    [1089.4943,	474.6895],
                    [2065.3442,  1460.2528],
                    [2071.0342,	482.1727]], dtype='float32')

#2022-06-03 @-5 deg EL
LED_REF_minus5 = np.array([
       [ 755.9869, 1394.8185],
       [ 755.    ,  638.0302],
       [1193.5675, 1827.2313],
       [1189.9103,  205.9062],
       [1942.25  , 1818.9421],
       [1936.6364,  210.4852],
       [2365.9902, 1385.127 ],
       [2362.5535, 640.5624]], dtype='float32')

LED_REF = LED_REF_minus5

PATTERN_LABEL_X_MIN = 1500
PATTERN_LABEL_X_MAX = 1900
PATTERN_LABEL_Y_MIN = 1500
PATTERN_LABEL_Y_MAX = 1850
PATTERN_CENTER_FROM_LABEL_BOUNDS = np.array(
    [(PATTERN_LABEL_X_MIN + PATTERN_LABEL_X_MAX) / 2., (PATTERN_LABEL_Y_MIN + PATTERN_LABEL_Y_MAX) / 2.])

# let's just hardcode pattern layout; useful for S1 alignment
DEFAULT_CENTROID_LAYOUT = np.array(
    [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
     1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
     1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
     1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
     1211, 1212, 1213, 1214,
     1311, 1312, 1313, 1314,
     1411, 1412, 1413, 1414,
     1111, 1112, 1113, 1114])

P1RY_OVERSHOOT_CENTROID_LAYOUT = np.array(
    [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
     1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
     1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
     1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
     1411, 1412, 1413, 1414,
     1111, 1112, 1113, 1114,
     1211, 1212, 1213, 1214,
     1311, 1312, 1313, 1314])

P2RY_OVERSHOOT_CENTROID_LAYOUT = np.array(
    [1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
     1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
     1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
     1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
     1211, 1212, 1213, 1214,
     1311, 1312, 1313, 1314,
     1411, 1412, 1413, 1414,
     1111, 1112, 1113, 1114])

RXm1_CENTROID_LAYOUT = np.array(
    [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
     1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
     1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
     1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
     1213, 1114, 1311, 1212, 1313, 1214, 1411, 1312, 1413, 1314, 1111, 1412, 1113, 1414, 1211, 1112])

RXm2_CENTROID_LAYOUT = np.array(
    [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
     1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
     1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
     1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
     1112, 1311, 1114, 1313, 1212, 1411, 1214, 1413, 1312, 1111, 1314, 1113, 1412, 1211, 1414, 1213])

NUM_VVV_DEFAULT = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

NUM_VVV_P1RY_OVERSHOOT = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             9, 10, 11, 12,
             13, 14, 15, 16,
             1, 2, 3, 4,
             5, 6, 7, 8
])

NUM_VVV_P2RY_OVERSHOOT = np.array(
            [17, 18, 19, 20, 21, 22, 23, 24,
             25, 26, 27, 28, 29, 30, 31, 32,
             1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

SEWPY_PARAMS = ["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUXERR_ISO", 'FLUX_AUTO', 'FLUXERR_AUTO', "FLUX_MAX", 'BACKGROUND',
                "KRON_RADIUS", "FLUX_RADIUS", "FLAGS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "CXX_IMAGE", "CYY_IMAGE",
                "CXY_IMAGE", "XMIN_IMAGE", "YMIN_IMAGE", "XMAX_IMAGE", "YMAX_IMAGE", "ELONGATION", "ELLIPTICITY",
                "ISOAREA_IMAGE", "ISOAREAF_IMAGE"]

VVV_COLS = ['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE', "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "CXX_IMAGE", "CYY_IMAGE",
            "CXY_IMAGE", "XMIN_IMAGE", "YMIN_IMAGE", "XMAX_IMAGE", "YMAX_IMAGE", 'FLUX_AREA', 'KRON_RADIUS', "FLUX_ISO",
            "FLUXERR_ISO", 'FLUX_AUTO', 'FLUXERR_AUTO', "FLUX_MAX", 'BACKGROUND', "ISOAREA_IMAGE", "ISOAREAF_IMAGE"]


def get_central_mod_corners(center=np.array([1891.25, 1063.75]),
                            cmod_xoffset=np.array([-109.25, -110.25, 108.75, 110.75]),
                            cmod_yoffset=np.array([111.25, -109.75, -111.75, 110.25])):
    x_corners = cmod_xoffset + center[0]
    y_corners = cmod_yoffset + center[1]
    return x_corners, y_corners


def get_perspective_transform_LEDs(LED_coords, led_ref=LED_REF):
    if not has_cv2:
        print("Can't do this, no CV2")
        return
    # perspective transformation matrix
    #pmat = cv2.getPerspectiveTransform(LED_REF, LED_coords)
    if len(LED_coords) == 4:
        pmat = cv2.getPerspectiveTransform(led_ref, LED_coords)
    elif len(LED_coords) == 8:
        pmat, status = cv2.findHomography(led_ref, LED_coords)
    else:
        print("Only found {} LEDs, can't do perspective transform".format(len(LED_coords)))
        return None
    return pmat

def get_CM_coords(LED_coords, led_ref=LED_REF):
    if not has_cv2:
        print("Can't do this, no CV2")
        return
    pmat = get_perspective_transform_LEDs(LED_coords, led_ref=led_ref)
    cm = np.zeros_like(CM_REF)
    if len(LED_coords) == 4:
        for i, c_ in enumerate(CM_REF):
            # only works for python3
            # cprime_ = cv2.perspectiveTransform(np.expand_dims(c_, axis=(0, 1)), pmat)[0,0]
            c_ = np.expand_dims(c_, axis=0)
            c_ = np.expand_dims(c_, axis=0)
            cprime_ = cv2.perspectiveTransform(c_, pmat)[0, 0]
            cm[i] = cprime_
    elif len(LED_coords) == 8:
        cm = cv2.warpPerspective(CM_REF, pmat, (2,4))
    return cm


def get_centroid_global(sew_out_table):
    xs_ = np.array(sew_out_table['X_IMAGE'], dtype=float)
    ys_ = np.array(sew_out_table['Y_IMAGE'], dtype=float)
    fs_ = np.array(sew_out_table['FLUX_ISO'], dtype=float)
    ind_ = np.where((xs_ > 1200) & (xs_ < 2500) & (ys_ > 400) & (ys_ < 1700))
    # print(xs_[ind_], ys_[ind_], fs_[ind_])
    xc = np.average(xs_[ind_], weights=fs_[ind_])
    yc = np.average(ys_[ind_], weights=fs_[ind_])
    print("==== Center of all centroids weighted by flux: {} {} ====".format(xc, yc))
    return xc, yc


def get_panel_position_in_pattern(panel_id, center=np.array([1891.25, 1063.75]), radius_mm=np.array([20, 40]),
                                  pixel_scale=0.241, phase_offset_rad=0, inner8=False):
    # in case of rx motion, we add a reference phase offset
    panel_id = str(panel_id)

    if list(panel_id)[0] == "1":
        is_primary = True
    else:
        is_primary = False
    if list(panel_id)[2] == "1":
        is_inner_ring = True
    else:
        is_inner_ring = False
    quadrant = float(list(panel_id)[1])
    if inner8:
        segment = float((int(list(panel_id)[3])-1)//2*2+1)
    else:
        segment = float(list(panel_id)[3])
    total_segment = 0.
    if is_primary and is_inner_ring:
        total_segment = 4.
    elif is_primary and not is_inner_ring:
        total_segment = 8.
    elif not is_primary and is_inner_ring:
        total_segment = 2.
    elif not is_primary and not is_inner_ring:
        total_segment = 4.
    phase = (quadrant - 1) * 0.5 * np.pi + 0.5 * np.pi * (segment - 0.5) / total_segment
    # Here we rotate by phase offset
    phase = phase + phase_offset_rad

    if is_inner_ring:
        radius = radius_mm[0]
        radius_pix = radius / pixel_scale
    else:
        radius = radius_mm[1]
        radius_pix = radius / pixel_scale
    cx = center[0] - radius_pix * np.sin(phase)
    cy = center[1] + radius_pix * np.cos(phase)

    return cx, cy, radius_pix, phase


def find_all_pattern_positions(all_panels=DEFAULT_CENTROID_LAYOUT, center=np.array([1891.25, 1063.75]),
                               radius_mm=np.array([20, 40]), phase_offset_rad=0,
                               # clockwise is positive, about 0.2 per outer panel
                               pixel_scale=0.241, outfile="dummy_pattern_position.txt", num_vvv=NUM_VVV_DEFAULT, inner8=False):
    # df_pattern = pd.DataFrame({'Panel': all_panels, '#': num_vvv})
    df_pattern = pd.DataFrame({'DefaultPanel': DEFAULT_CENTROID_LAYOUT, 'Panel': all_panels, '#': num_vvv})
    df_pattern['Xpix'] = 0
    df_pattern['Ypix'] = 0
    df_pattern['Rpix'] = 0
    df_pattern['Phase'] = 0
    for i, row in df_pattern.iterrows():
        # x_, y_, r_pix_, phase_ = get_panel_position_in_pattern(row['Panel'], center=center, radius_mm=radius_mm,
        x_, y_, r_pix_, phase_ = get_panel_position_in_pattern(row['DefaultPanel'], center=center, radius_mm=radius_mm,
                                                               phase_offset_rad=phase_offset_rad,
                                                               pixel_scale=pixel_scale, inner8=inner8)
        df_pattern.loc[i, 'Xpix'] = x_
        df_pattern.loc[i, 'Ypix'] = y_
        df_pattern.loc[i, 'Rpix'] = r_pix_
        df_pattern.loc[i, 'Phase'] = phase_

    df_pattern = df_pattern[['Panel', '#', 'Xpix', 'Ypix', 'Rpix', 'Phase']]
    df_pattern.to_csv(outfile, index=False, sep="\t")
    return df_pattern


def read_raw(f='./GAS_image.raw', cols=2592, rows=1944, outfile=None, show=False):
    fd = open(f, 'rb')
    f = np.fromfile(fd, dtype=np.uint8, count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    fd.close()
    if outfile is not None:
        if has_cv2:
            cv2.imwrite(outfile, im)
        else:
            plt.imshow(im, cmap='gray')
            plt.savefig(outfile)
    if show:
        plt.imshow(im, cmap='gray')
    return im


def read_png(f, show=False):
    im = cv2.imread(f)
    if show:
        plt.imshow(im, cmap='gray')
    return im


def read_image(f, cols=2592, rows=1944, outfile=None, show=False):
    if os.path.splitext(f)[1] == ".raw":
        return read_raw(f, cols=cols, rows=rows, outfile=outfile, show=show)
    elif os.path.splitext(f)[1] == ".png":
        return read_png(f, show=show)
    else:
        print("Image extension not found for path {}, printing null matrix. ".format(f))
        return np.zeros((rows, cols))


def raw2fits(f='./GAS_image.raw', cols=2592, rows=1944, outfile=None):
    from astropy.io import fits
    fd = open(f, 'rb')
    fdata = np.fromfile(fd, dtype=np.uint8, count=rows * cols)
    im = fdata.reshape((rows, cols))  # notice row, column format
    fd.close()
    fitf = fits.PrimaryHDU(data=im)

    if outfile is None:
        outfile = f.split(".")[0] + ".fits"
    fitf.writeto(outfile)

    # return im


def im2fits(im, outfile, overwrite=True):
    from astropy.io import fits
    fitf = fits.PrimaryHDU(data=im)
    fitf.writeto(outfile, overwrite=overwrite)  # return im


def crop_image_ellipse(im_best, df_best, i, r_ellipse, sub_bkg=True, verbose=False):
    xmin = int(df_best.XMIN_IMAGE[i] * 0.99)
    xmax = int(df_best.XMAX_IMAGE[i] * 1.01)
    ymax = int(df_best.YMIN_IMAGE[i] * 0.99)
    ymin = int(df_best.YMAX_IMAGE[i] * 1.01)

    cxx = df_best.CXX_IMAGE[i]
    cyy = df_best.CYY_IMAGE[i]
    cxy = df_best.CXY_IMAGE[i]
    xmean = df_best.X_IMAGE[i]
    ymean = df_best.Y_IMAGE[i]
    R = r_ellipse

    h, w = np.shape(im_best)
    x = range(w)
    y = range(h)
    X, Y = np.meshgrid(x, y)
    ellipse_mask = cxx * (X - xmean) ** 2 + cyy * (Y - ymean) ** 2 + cxy * (X - xmean) * (Y - ymean) > R ** 2

    if verbose:
        print("Cropping for panel {}: XMIN={}, XMAX={}, YMIN={}, YMAX={}".format(df_best.Panel_ID_guess[i], xmin, xmax,
                                                                                 ymin, ymax))
    if sub_bkg:
        im_best = im_best - df_best['BACKGROUND'][i]

    im_best[ellipse_mask] = 0
    im_crop = im_best[ymax:ymin, xmin:xmax]

    return im_crop


def crop_image(im_best, df_best, i, sub_bkg=True, verbose=False):
    xmin = int(df_best.XMIN_IMAGE[i])
    xmax = int(df_best.XMAX_IMAGE[i])
    ymax = int(df_best.YMIN_IMAGE[i])
    ymin = int(df_best.YMAX_IMAGE[i])
    if verbose:
        print("Cropping for panel {}: XMIN={}, XMAX={}, YMIN={}, YMAX={}".format(df_best.Panel_ID_guess[i], xmin, xmax,
                                                                                 ymin, ymax))
    im_crop = im_best[ymax:ymin, xmin:xmax]
    if sub_bkg:
        im_crop = im_crop - df_best['BACKGROUND'][i]
    return im_crop


def crop_rotate_image(im_best, df_best, i, sub_bkg=True, verbose=False, reshape=True):
    from scipy import ndimage  # , misc
    img = crop_image(im_best, df_best, i, sub_bkg=sub_bkg)
    # img = img-['BACKGROUND'][0]
    img_rotated = ndimage.rotate(img, df_best['THETA_IMAGE'][i], reshape=reshape)
    if verbose:
        print("Rotating for panel {}: THETA={}".format(df_best.Panel_ID_guess[i], df_best['THETA_IMAGE'][i]))

    return img_rotated


def image_statistics_2D(Z):
    h, w = np.shape(Z)

    x = range(w)
    y = range(h)

    X, Y = np.meshgrid(x, y)

    # Centroid (mean)
    m00 = np.sum(Z)
    cx = np.sum(Z * X) / m00
    cy = np.sum(Z * Y) / m00

    dx = (x - cx)
    dy = (y - cy)
    DX, DY = np.meshgrid(dx, dy)

    # First Momentum
    m11 = np.sum(Z * DX) / m00
    m12 = np.sum(Z * DY) / m00

    # Second Momentum
    X2 = DX * DX
    Y2 = DY * DY
    XY = DX * DY

    m20 = np.sum(Z * (X2 + Y2)) / m00
    m21 = np.sum(Z * (X2 - Y2)) / m00
    m22 = np.sum(Z * (2 * XY)) / m00

    # Third Momentum
    X3 = DX * DX * DX
    Y3 = DY * DY * DY
    X2Y = DX * DX * DY
    Y2X = DY * DY * DX

    # Find the third central moments
    m31 = np.sum(Z * (X3 + Y2X)) / m00
    m32 = np.sum(Z * (X2Y + Y3)) / m00
    m33 = np.sum(Z * (Y3 - 3 * X2Y)) / m00
    m34 = np.sum(Z * (X3 - 3 * Y2X)) / m00

    # Fourth Momentum
    X4 = DX * DX * DX * DX
    Y4 = DY * DY * DY * DY
    XY3 = DX * DY * DY * DY
    X3Y = DX * DX * DX * DY
    X2Y2 = DX * DX * DY * DY

    # Find the fourth central moments
    m40 = np.sum(Z * (X2 + Y2) ** 2) / m00
    m41 = np.sum(Z * (X4 - Y4)) / m00
    m42 = np.sum(Z * (2 * XY3 + 2 * X3Y)) / m00
    m43 = np.sum(Z * (X4 - 6 * X2Y2 + Y4)) / m00
    m44 = np.sum(Z * (4 * X3Y - 4 * XY3)) / m00

    return cx, cy, m00, m11, m12, m20, m21, m22, m31, m32, m33, m34, m40, m41, m42, m43, m44


def report_moments(img):
    # Calculate the image statistics using the projection method
    # stats_pr = image_statistics(img)

    # Confirm that they are the same by using a 2D calculation
    stats_2d = image_statistics_2D(img)

    names = ('Centroid x', 'Centroid y', 'StdDev x', 'StdDev y', 'sqrt cov xy', 'm00', 'm30', 'm03', 'm21', 'm12')

    # print('Statistis\t1D\t2D')
    # for name,i1,i2 in zip(names, stats_2d,stats_pr):
    #    print('%s \t%.2f \t%.2f'%(name, i1,i2))
    # for name,i1 in zip(names, stats_pr):
    for name, i1 in zip(names, stats_2d):
        print('%s \t%.2f ' % (name, i1))


def comp_moments(img, img2):
    # Calculate the image statistics using the projection method
    # stats_pr = image_statistics(img)
    # stats_pr2 = image_statistics(img2)

    # Confirm that they are the same by using a 2D calculation
    stats_2d = image_statistics_2D(img)
    stats_2d2 = image_statistics_2D(img2)

    names = ('Centroid x', 'Centroid y', 'StdDev x', 'StdDev y', 'sqrt cov xy', 'm00', 'm30', 'm03', 'm21', 'm12')

    print('============== \tcropped\trotated ')
    # for name,i1,i2 in zip(names, stats_pr,stats_pr2):
    for name, i1, i2 in zip(names, stats_2d, stats_2d2):
        print('%s \t%.2f \t%.2f' % (name, i1, i2))


def get_skewness(im1, df1, r_ellipse, pind=9, show=False, verbose=True, reshape=True):
    # img_cropped = crop_image(im1, df1, pind, verbose=True)
    img_rotated = crop_rotate_image(im1, df1, pind, verbose=True, reshape=reshape)
    # img_rotated = crop_rotate_image(im1, df1, pind, verbose=True)
    img_cropped_ellipse = crop_image_ellipse(im1, df1, pind, r_ellipse, verbose=True)

    # comp_moments(img_cropped, img_rotated)
    stats_2d = image_statistics_2D(img_cropped_ellipse)
    stats_dict = {}
    names = (
        'cx', 'cy', 'm00', 'm11', 'm12', 'm20', 'm21', 'm22', 'm31', 'm32', 'm33', 'm34', 'm40', 'm41', 'm42', 'm43',
        'm44')
    for name, i1 in zip(names, stats_2d):
        stats_dict[name] = i1
    A0 = df1.A_IMAGE[pind]
    A = np.sqrt(0.5 * (stats_dict['m20'] + np.sqrt(stats_dict['m21'] ** 2 + stats_dict['m22'] ** 2)))
    epsA = (A0 - A) / A0
    B0 = df1.B_IMAGE[pind]
    B = np.sqrt(0.5 * (stats_dict['m20'] - np.sqrt(stats_dict['m21'] ** 2 + stats_dict['m22'] ** 2)))
    epsB = (B0 - B) / B0
    theta0 = df1.THETA_IMAGE[pind]
    theta = 0.5 * np.rad2deg(np.arctan2(stats_dict['m22'], stats_dict['m21']))

    stats_dict['cx'] = stats_dict['cx'] + df1.XMIN_IMAGE[pind] * 0.99
    stats_dict['cy'] = stats_dict['cy'] + df1.YMIN_IMAGE[pind] * 0.99

    if verbose:
        for name, i1 in sorted(stats_dict.items()):
            print('%s \t%.2f ' % (name, i1))
        print("A: \t%.2f (%.2f)" % (A, A0))
        print("epsA: \t%0.3f" % epsA)
        print("B: \t%.2f (%.2f)" % (B, B0))
        print("epsB: \t%0.3f" % epsB)
        print("elongation \t%.2f (%.2f)" % ((A / B), (A0 / B0)))
        print("ellipticity \t%.2f (%.2f)" % (1 - (B / A), 1 - (B0 / A0)))
        print("theta: \t%.2f (%.2f)" % (theta, theta0))

    if show:
        fig = plt.figure(figsize=(7, 3))
        ax1, ax3 = fig.subplots(1, 2)
        img_cropped = crop_image(im1, df1, pind, verbose=True)
        ax1.imshow(img_cropped, cmap='gray')
        ax1.set_axis_off()
        # ax2.imshow(img_rotated, cmap='gray')
        # ax2.set_axis_off()
        ax3.imshow(img_cropped_ellipse, cmap='gray')
        ax3.set_axis_off()
        fig.set_tight_layout(True)
        plt.show()

    return stats_dict


def plot_sew_cat(dst_trans, sew_out_trans, brightestN=0, xlim=None, ylim=None, outfile=None, show=False, vmax=None,
                 pattern_label_x_min=0, pattern_label_x_max=0, pattern_label_y_min=0, pattern_label_y_max=0,
                 df_LEDs=None, center_offset=[0, 0]):
    '''
    Plots image with imshow. On top, plots only sources in sew_out_trans DF within the pattern search region.
    '''
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    if vmax is not None:
        ax_img = ax.imshow(dst_trans, vmax=vmax, cmap='gray')
    else:
        ax_img = ax.imshow(dst_trans, cmap='gray')
    fig.colorbar(ax_img)

    i = 0
    for row in sew_out_trans['table']:
        i += 1
        if brightestN > 0 and i > brightestN:
            break
        # print(row)

        e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2, height=row['B_IMAGE'] * 2,
                    angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('r')
        ax.add_artist(e)

        if pattern_label_x_max > 0 and pattern_label_y_max > 0 and row['X_IMAGE'] <= pattern_label_x_max and row[
            'X_IMAGE'] >= pattern_label_x_min and row['Y_IMAGE'] >= pattern_label_y_min and row[
            'Y_IMAGE'] <= pattern_label_y_max:
            center_pattern = np.array(
                [(pattern_label_x_min + pattern_label_x_max) / 2., (pattern_label_y_min + pattern_label_y_max) / 2.])

            ax.annotate(int(row['ID']), xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data', xytext=(
                np.array([row['X_IMAGE'] + row['X_IMAGE'] - center_pattern[0],
                          row['Y_IMAGE'] + row['Y_IMAGE'] - center_pattern[1]])),  # for orig lens
                        # xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                        color='c', alpha=0.8,
                        arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                        alpha=0.7), )
            e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2,
                        height=row['B_IMAGE'] * 2, angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_color('c')

    if df_LEDs is not None:
        if len(df_LEDs) == 4 or len(df_LEDs) == 8:
            for i, row in df_LEDs.iterrows():
                e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2,
                            height=row['B_IMAGE'] * 2, angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('gold')
                ax.add_artist(e)
            center = [np.mean(df_LEDs['X_IMAGE']) + center_offset[0], np.mean(df_LEDs['Y_IMAGE']) + center_offset[1]]
            plt.plot([center[0]], [center[1]], color='gold', marker='+')

            # plot the CM after perspective transform
            LED_coords = df_LEDs[['X_IMAGE', 'Y_IMAGE']].to_numpy(dtype='float32')
            print("getting CM coords")
            cm = get_CM_coords(LED_coords)
            print(cm)
            print("plotting CM coords")
            for i, row in cm:
                cm_center = np.mean(cm, axis=0)
                ax.plot([cm_center[0]], [cm_center[1]], 'g+', alpha=0.4)
                ax.scatter(cm[:, 0], cm[:, 1], s=2, facecolors='none', edgecolors='g', alpha=0.4)

    zoom = False
    if xlim is not None:
        plt.xlim(xlim)
        zoom = True
    if ylim is not None:
        plt.ylim(ylim)
        zoom = True

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile)
    if show:
        plt.show()


def calc_ring_pattern(sewtable, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM, rad_frac=0.5,
                      weight=False, fix_center=True):
    '''
    Draw a circle centered at some point and select sources within it, then get distance for those sources to
    the center. For circular region, use pattern_center and given radius. If not circle, use entire sewtable and given center.
    With this cut, get averaged center of sources, mean and std of radius, and sliced sewtable.
    '''

    if rad_frac > 1 or rad_frac < 0 or radius < 0:
        print("Params to find ring pattern is not sensible")
        return

    if not fix_center:
        # figure out center here
        xlo = (pattern_center[0] - (1 + rad_frac) * radius)
        xhi = (pattern_center[0] + (1 + rad_frac) * radius)
        ylo = (pattern_center[1] - (1 + rad_frac) * radius)
        yhi = (pattern_center[1] + (1 + rad_frac) * radius)

        sew_slice = sewtable[
            (sewtable['X_IMAGE'] <= xhi) & (sewtable['X_IMAGE'] >= xlo) & (sewtable['Y_IMAGE'] <= yhi) & (
                    sewtable['Y_IMAGE'] >= ylo)]

        if weight:
            centroidx = np.average(sew_slice['X_IMAGE'], weights=sew_slice['FLUX_ISO'])
            centroidy = np.average(sew_slice['Y_IMAGE'], weights=sew_slice['FLUX_ISO'])
        else:
            centroidx = np.average(sew_slice['X_IMAGE'])
            centroidy = np.average(sew_slice['Y_IMAGE'])
    else:
        # don't figure out center, take the input
        centroidx = pattern_center[0]
        centroidy = pattern_center[1]
        sew_slice = sewtable

    r2s = (sew_slice['X_IMAGE'] - centroidx) ** 2 + (sew_slice['Y_IMAGE'] - centroidy) ** 2

    # iterate
    sew_slice['R_pattern'] = np.sqrt(r2s)
    # print(sew_slice)
    sew_slice = sew_slice[
        (sew_slice['R_pattern'] <= (radius * (1 + rad_frac))) & (sew_slice['R_pattern'] >= (radius * (1 - rad_frac)))]
    # print("after")
    # print(sew_slice)

    if weight:
        centroidx = np.average(sew_slice['X_IMAGE'], weights=sew_slice['FLUX_ISO'])
        centroidy = np.average(sew_slice['Y_IMAGE'], weights=sew_slice['FLUX_ISO'])
    else:
        centroidx = np.average(sew_slice['X_IMAGE'])
        centroidy = np.average(sew_slice['Y_IMAGE'])

    r2s = (sew_slice['X_IMAGE'] - centroidx) ** 2 + (sew_slice['Y_IMAGE'] - centroidy) ** 2
    phase = np.arcsin((centroidx - sew_slice['X_IMAGE']) / sew_slice['R_pattern'])
    sew_slice['Phase'] = phase

    r2mean = np.mean(r2s)
    # r2std = np.std(r2s)
    r2std = np.std(r2s) / (len(r2s) * 1.0)
    print(len(r2s), r2s.shape)
    print("r2std={}, r2std/(N)={}".format(r2std, r2std * (len(r2s) * 1.0)))

    new_center = np.array([centroidx, centroidy])
    new_radius = np.sqrt(r2mean)

    return r2mean, r2std, new_center, new_radius, sew_slice


def calc_ring_pattern_band(sewtable, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM, rad_inner=0.5,
                           rad_outer=1.2, weight=False):
    # this is just a stupid calc func, no optimization / minimization
    if radius < 0 or rad_inner > 1 or rad_outer < 1:
        print("Params to find ring pattern is not sensible")
        return
    xlo = (pattern_center[0] - (rad_outer) * radius)
    xhi = (pattern_center[0] + (rad_outer) * radius)
    ylo = (pattern_center[1] - (rad_outer) * radius)
    yhi = (pattern_center[1] + (rad_outer) * radius)

    # pass 1 just cut a square
    sew_slice = sewtable[(sewtable['X_IMAGE'] <= xhi) & (sewtable['X_IMAGE'] >= xlo) & (sewtable['Y_IMAGE'] <= yhi) & (
            sewtable['Y_IMAGE'] >= ylo)]

    if weight:
        centroidx = np.average(sew_slice['X_IMAGE'], weights=sew_slice['FLUX_ISO'])
        centroidy = np.average(sew_slice['Y_IMAGE'], weights=sew_slice['FLUX_ISO'])
    else:
        centroidx = np.average(sew_slice['X_IMAGE'])
        centroidy = np.average(sew_slice['Y_IMAGE'])

    r2s = (sew_slice['X_IMAGE'] - centroidx) ** 2 + (sew_slice['Y_IMAGE'] - centroidy) ** 2

    # iterate; pass 2 cut annulus
    sew_slice['R_pattern'] = np.sqrt(r2s)
    # print(sew_slice)
    # slice changed
    sew_slice = sew_slice[
        (sew_slice['R_pattern'] <= (radius * (rad_inner))) & (sew_slice['R_pattern'] >= (radius * (rad_outer)))]
    # print("after")
    # print(sew_slice)

    # centroid changed
    if weight:
        centroidx = np.average(sew_slice['X_IMAGE'], weights=sew_slice['FLUX_ISO'])
        centroidy = np.average(sew_slice['Y_IMAGE'], weights=sew_slice['FLUX_ISO'])
    else:
        centroidx = np.average(sew_slice['X_IMAGE'])
        centroidy = np.average(sew_slice['Y_IMAGE'])

    # Rs changed; pass 3
    r2s = (sew_slice['X_IMAGE'] - centroidx) ** 2 + (sew_slice['Y_IMAGE'] - centroidy) ** 2
    sew_slice['R_pattern'] = np.sqrt(r2s)
    # slice changed again
    sew_slice = sew_slice[
        (sew_slice['R_pattern'] <= (radius * (rad_inner))) & (sew_slice['R_pattern'] >= (radius * (rad_outer)))]

    # centroid changed again
    if weight:
        centroidx = np.average(sew_slice['X_IMAGE'], weights=sew_slice['FLUX_ISO'])
        centroidy = np.average(sew_slice['Y_IMAGE'], weights=sew_slice['FLUX_ISO'])
    else:
        centroidx = np.average(sew_slice['X_IMAGE'])
        centroidy = np.average(sew_slice['Y_IMAGE'])

    r2mean = np.mean(r2s)
    r2std = np.std(r2s) / (len(r2s) * 1.0)
    print("r2std={}, r2std/sqrt(N)={}".format(r2std, r2std * (len(r2s) * 1.0)))
    newc = np.array([centroidx, centroidy])
    newr = np.sqrt(r2mean)

    return r2mean, r2std, newc, newr, sew_slice  # note that returned sew_slice may be of limited use


def radii_clustering(sewtable, n_rings=2, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM,
                     rad_inner=0.5, rad_outer=1.2, weight=False):
    # implementing
    # filtering
    if radius < 0 or rad_inner > 1 or rad_outer < 1:
        print("Params to find ring pattern is not sensible")
        return

    # lets first treat the pattern center is legit
    # don't figure out center, take the input
    centroidx = pattern_center[0]
    centroidy = pattern_center[1]
    sew_slice = sewtable

    r2s = (sew_slice['X_IMAGE'] - centroidx) ** 2 + (sew_slice['Y_IMAGE'] - centroidy) ** 2
    sew_slice['R_pattern'] = np.sqrt(r2s)
    sew_slice = sew_slice[
        (sew_slice['R_pattern'] <= (radius * (rad_inner))) & (sew_slice['R_pattern'] >= (radius * (rad_outer)))]

    kmeans_ = KMeans(n_clusters=n_rings)
    clusters_ = kmeans_.fit_predict(sew_slice['R_pattern'])  # clustering of radii^2

    sew_slice['Ring_ID'] = clusters_

    rad_means = []
    rad_stds = []

    for i in range(n_rings):
        this_group = sew_slice[(sew_slice['Ring_ID'] == i)]
        this_rs = this_group['R_pattern']
        rad_means.append(np.mean(this_rs))
        rad_stds.append(np.std(this_rs))

    return sew_slice, rad_means, rad_stds


def find_ring_pattern_clustering(sewtable, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM,
                                 rad_frac=0.2, rad_tol_frac=0.1, n_rings=2, rad_inner=0.5, rad_outer=1.2,
                                 rad_edges_pix=[50, 110, 150, 200], phase_offset=0, n_iter=50, chooseinner=False,
                                 chooseouter=False, tryouter=True, ring_tol_rms=400):
    # implementing
    """
    thoughts: we need to find imperfect rings, up to 3 rings, 1) 16 centroids from P1-S1 (bright); 2) 32 from P2-S1 (should be dimmer); 32 from P2-S2
        when pointing off-axis there will be more ghosts...
    The reason for imperfection could be 1) alignment has been done using MPESs 2) panel motion executed in order to find matrices for rings altogether
    These rings are supposed to have different radii, but may be close together to keep things in center near nominal position for better focus;
        givent this, we could try to search for rings in annulus, and relax the ring rms cut, better include more than not enough...
        note that the calc_ring_part do not optimize, have to handle it here...
    We also attempt to guess panel id based on phase; this is quite easy for the nominal pattern position; but, when Rx motion introduced this will all mess up;
        the good thing is that the relative phase of all panels will be approx. preserved, so we should introduce a phase_offset param
        (1221 / 1211 has reference phase of 0.53pi in picture)

    """
    if rad_frac > 1 or rad_frac < 0 or radius < 0 or n_rings > 3 or n_rings < 0:
        print("Params to find ring pattern is not sensible")
        return

    print("Doing clustering")
    good_ring = False

    sew_slice, rad_means, rad_stds = radii_clustering(sewtable, n_rings=n_rings, pattern_center=pattern_center,
                                                      radius=radius, rad_inner=rad_inner, rad_outer=rad_outer, )
    print("R Variance not decreasing anymore on the {}th iteration.".format(i))
    print("Rvar/N = {}".format(r2std_last / len(sew_slice)))
    if rad_stds / len(sew_slice) < 400 and len(sew_slice) > 4:
        print("This seems to be a pretty good ring")
        good_ring = True
    else:
        print("Crappy ring")

    print(rad_means, rad_stds)

    return
    if good_ring:
        if abs(len(sew_slice) - 16) <= abs(len(sew_slice) - 32) or chooseinner:
            # guess this is inner ring
            df_pattern = find_all_pattern_positions(center=clast,
                                                    radius_mm=np.array([rlast * 0.241, rlast * 2 * 0.241]),
                                                    pixel_scale=0.241, outfile=None, )
            print("Found {} candidate centroid forming an inner ring".format(len(sew_slice)))
            print("Center {}, radius {}".format(clast, rlast))
            df_slice = df_pattern[(abs(df_pattern.Rpix - rlast) < (rlast * rad_tol_frac))]
            print(
                "After applying a tolerance fraction cut = {}, {} panels left ".format(rad_tol_frac, df_slice.shape[0]))
            if tryouter:
                df_outer_slice = df_pattern[(abs(df_pattern.Rpix - 2 * rlast) < (2 * rlast * rad_tol_frac))]
                df_slice.append(df_outer_slice)
        elif abs(len(sew_slice) - 16) > abs(len(sew_slice) - 32) or chooseouter:
            # not tested
            df_pattern = find_all_pattern_positions(center=clast,
                                                    radius_mm=np.array([rlast / 2 * 0.241, rlast * 0.241]),
                                                    pixel_scale=0.241, outfile=None, )
            df_slice = df_pattern[(abs(df_pattern.Rpix - rlast) < (rlast * rad_tol_frac))]
    else:
        df_slice = None

    return clast, rlast, r2std_last, sew_slice, df_slice


def find_ring_pattern(sewtable, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM, rad_frac=0.2,
                      rad_tol_frac=0.1, n_iter=50, chooseinner=False, chooseouter=False, tryouter=True, fix_center=True,
                      phase_offset_rad=0, get_center=False, var_tol=400, all_panels=DEFAULT_CENTROID_LAYOUT, num_vvv=NUM_VVV_DEFAULT, inner8=False):
    '''
    Use sewtable of sources to find the mean centroid of the sources. Start with some radius and get the sources within that
    radius, their mean/std of their distances to center, and their mean center. Iterate with new radius cuts and new
    source selections. At the minimum std of the distances to center, call that a good_ring. For this good ring, apply
    Panel_ID label to each source in the corresponding locations.

    '''
    if rad_frac > 1 or rad_frac < 0 or radius < 0:
        print("Params to find ring pattern is not sensible")
        return
    r2std_last = 1e9
    last_center = pattern_center
    last_radius = radius
    good_ring = False
    df_slice = None

    centroidx = np.average(sewtable['X_IMAGE'])  # , weights=sewtable['FLUX_ISO'])
    centroidy = np.average(sewtable['Y_IMAGE'])  # , weights=sewtable['FLUX_ISO'])

    if np.sqrt((centroidx - pattern_center[0]) ** 2 + (centroidy - pattern_center[1]) ** 2) > 100:
        print("Your center is >100 pixels away from the flux-weighted center of gravity of all centroids")
        print("Flux-weighted center of gravity of all centroids is X={}, Y={}".format(centroidx, centroidy))

    for i in range(n_iter):
        r2mean, r2std, new_center, new_radius, sew_slice = calc_ring_pattern(sewtable, pattern_center=last_center,
                                                                             radius=last_radius, rad_frac=rad_frac,
                                                                             fix_center=fix_center)
        if len(sew_slice) == 0:
            print("no centroid found for this ring")
            break
        if r2std < r2std_last:
            last_center = new_center
            last_radius = new_radius
            r2std_last = r2std
        else:
            print("R Variance not decreasing anymore on the {}th iteration.".format(i))
            print("Rvar/N = {}".format(r2std_last / len(sew_slice)))
            print("R = {} pix, center(x,y) = {}, {}".format(last_radius, last_center[0], last_center[1]))
            if r2std_last / len(sew_slice) < var_tol and len(sew_slice) > 4 and not get_center:
                print("Found {} panels on this ring".format(len(sew_slice)))
                print("This seems to be a pretty good ring")
                good_ring = True
            elif r2std_last / len(sew_slice) < var_tol:
                print("Found {} panels on this ring".format(len(sew_slice)))
                print("This seems to be a pretty good ring")
                good_ring = True
                continue
            else:
                print("Crappy ring")  # break

    if good_ring:
        if abs(len(sew_slice) - 16) + 4 <= abs(len(sew_slice) - 32) or chooseinner:
            # guess this is inner ring
            df_pattern = find_all_pattern_positions(all_panels=all_panels, center=last_center,
                                                    radius_mm=np.array([last_radius * 0.241, last_radius * 2 * 0.241]),
                                                    pixel_scale=0.241, phase_offset_rad=phase_offset_rad,
                                                    outfile=None, num_vvv=num_vvv, inner8=inner8)
            print("Found {} candidate centroid forming an inner ring".format(len(sew_slice)))
            print("Center {}, radius {}".format(last_center, last_radius))
            df_slice = df_pattern[(abs(df_pattern.Rpix - last_radius) < (last_radius * rad_tol_frac))]
            print(
                "After applying a tolerance fraction cut = {}, {} panels left ".format(rad_tol_frac, df_slice.shape[0]))
            if tryouter:
                df_outer_slice = df_pattern[(abs(df_pattern.Rpix - 2 * last_radius) < (2 * last_radius * rad_tol_frac))]
                df_slice.append(df_outer_slice)
        elif abs(len(sew_slice) - 16) > abs(len(sew_slice) - 32) or chooseouter:
            # not tested
            df_pattern = find_all_pattern_positions(all_panels=all_panels, center=last_center,
                                                    radius_mm=np.array([last_radius / 2 * 0.241, last_radius * 0.241]),
                                                    pixel_scale=0.241, phase_offset_rad=phase_offset_rad,
                                                    outfile=None, num_vvv=num_vvv)
            print("Found {} candidate centroid forming an outer ring".format(len(sew_slice)))
            print("Center {}, radius {}".format(last_center, last_radius))
            df_slice = df_pattern[(abs(df_pattern.Rpix - last_radius) < (last_radius * rad_tol_frac))]
            print(
                "After applying a tolerance fraction cut = {}, {} panels left ".format(rad_tol_frac, df_slice.shape[0]))
    else:
        df_slice = None

    return last_center, last_radius, r2std_last, sew_slice, df_slice


# def find_LEDs(sewtable, coords=[[1385, 590], [1377, 1572], [2360, 1576], [2365, 597]],
#def find_LEDs(sewtable, coords=[[1095, 500], [1087, 1472], [2070, 1476], [2075, 497]], search_width_x=20,
#def find_LEDs(sewtable, coords=[[1090, 475], [1087, 1460], [2065, 1460], [2071, 482]], search_width_x=40,
def find_LEDs(sewtable, coords=[
       [ 755.9869, 1394.8185],
       [ 755.    ,  638.0302],
       [1193.5675, 1827.2313],
       [1189.9103,  205.9062],
    [1942.25, 1818.9421],
    [1936.6364, 210.4852],
    [2365.9902, 1385.127],
    [2362.5535,  640.5624]], search_width_x=40,
              search_width_y=20, center_offset=[0, 0]):
    df_out = pd.DataFrame()
    N_LEDs = len(coords)
    for i, c_ in enumerate(coords):
        x, y = c_[0], c_[1]
        xmin = x - search_width_x
        xmax = x + search_width_x
        ymin = y - search_width_y
        ymax = y + search_width_y
        df_ = sewtable[(sewtable['X_IMAGE'] <= xmax) & (sewtable['X_IMAGE'] >= xmin) & (sewtable['Y_IMAGE'] <= ymax) & (
                    sewtable['Y_IMAGE'] >= ymin)].to_pandas()
        # print(i, x,y)
        # print(df_)
        # if df_out.empty:
        #    df_out = df_
        # else:
        df_out = df_out.append(df_)
    # print(df_out)
    if len(df_out) > N_LEDs:
        print("==== {} LEDs found, more than expected {} ====".format(len(df_out), N_LEDs))
        print("Will only keep the brightest {}".format(len(df_out)))
        df_out = df_out.nlargest(N_LEDs, 'FLUX_ISO')
        center = [np.mean(df_out['X_IMAGE']) + center_offset[0], np.mean(df_out['Y_IMAGE']) + center_offset[1]]
        print("==== Center of the LEDs is {:.2f}, {:.2f} ====".format(center[0], center[1]))
    elif len(df_out) == N_LEDs:
        print("==== All {} LEDs found ====".format(N_LEDs))
        center = [np.mean(df_out['X_IMAGE']) + center_offset[0], np.mean(df_out['Y_IMAGE']) + center_offset[1]]
        print("==== Center of the LEDs is {:.2f}, {:.2f} ====".format(center[0], center[1]))
    else:
        print("==== *** Only {} LEDs found out of {}!!! *** ====".format(len(df_out), N_LEDs))
        center = [0, 0]

    # df_out['X100'] = df_out['X_IMAGE']//100*100
    # df_out['Y100'] = df_out['Y_IMAGE']//100*100
    df_out['X100'] = df_out['X_IMAGE'].apply(lambda x: int(round(x / 100.0)) * 100)
    df_out['Y100'] = df_out['Y_IMAGE'].apply(lambda x: int(round(x / 100.0)) * 100)
    # df_out = df_out.sort_values(['X_IMAGE', 'Y_IMAGE'], ascending=[True, False])
    df_out = df_out.sort_values(['X100', 'Y100'], ascending=[True, False])
    df_out = df_out.drop(columns=['X100', 'Y100'])
    df_out['ID'] = range(1, len(df_out) + 1)
    return df_out, center


def find_single_ring_pattern(sewtable, pattern_center=PATTERN_CENTER_FROM_LABEL_BOUNDS, radius=20 / PIX2MM,
                             rad_frac=0.2, rad_tol_frac=0.1, n_iter=50, num_vvv=NUM_VVV_DEFAULT):
    # implementing, may dropt it
    if rad_frac > 1 or rad_frac < 0 or radius < 0:
        print("Params to find ring pattern is not sensible")
        return
    r2std_last = 1e9
    clast = pattern_center
    rlast = radius
    good_ring = False
    for i in range(n_iter):
        r2mean, r2std, newc, newr, sew_slice = calc_ring_pattern(sewtable, pattern_center=clast, radius=rlast,
                                                                 rad_frac=rad_frac)
        if r2std < r2std_last:
            clast = newc
            rlast = newr
            r2std_last = r2std
        else:
            print("R Variance not decreasing anymore on the {}th iteration.".format(i))
            print("Rvar/N = {}".format(r2std_last / len(sew_slice)))
            if r2std_last / len(sew_slice) < 400 and len(sew_slice) > 4:
                print("This seems to be a pretty good ring")
                good_ring = True
            else:
                print("Crappy ring")
            break

    if good_ring:
        df_pattern = find_all_pattern_positions(center=clast, radius_mm=np.array([rlast * 0.241, rlast * 2 * 0.241]),
                                                pixel_scale=0.241, outfile=None, num_vvv=num_vvv )
        print("Found {} candidate centroid forming an inner ring".format(len(sew_slice)))
        print("Center {}, radius {}".format(clast, rlast))
        df_slice = df_pattern[(abs(df_pattern.Rpix - rlast) < (rlast * rad_tol_frac))]
        print("After applying a tolerance fraction cut = {}, {} panels left ".format(rad_tol_frac, df_slice.shape[0]))
        if tryouter:
            df_outer_slice = df_pattern[(abs(df_pattern.Rpix - 2 * rlast) < (2 * rlast * rad_tol_frac))]
            df_slice.append(df_outer_slice)
    else:
        df_slice = None

    return clast, rlast, r2std_last, sew_slice, df_slice


def process_raw(rawfile, kernel_w=3, DETECT_MINAREA=30, THRESH=5, DEBLEND_MINCONT=0.02, BACK_SIZE=128, clean=True,
                sewpy_params=SEWPY_PARAMS, cropxs=(1350, 1800), cropys=(1250, 800), savecatalog_name=None,
                savefits_name=None, overwrite_fits=True, saveplot_name=None, show=False, search_xs=[0, 0], df_LEDs=None,
                search_ys=[0, 0]):
    '''
    This actually processes the file with sewpy and extracts sources. The 'rawfile' is a matrix (not the path to the .RAW image).
    After processing with sewpy, it pushes the extracted sources object to a Pandas dataframe, then cleans for flags and search region.
    With the remaining list, it plots the image and ellipse labels with `plot_sew_cat`.
    '''
    from astropy.table import Column

    im_raw = read_image(rawfile)

    if has_cv2:
        median = cv2.medianBlur(im_raw, kernel_w)
    else:
        print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
        median = im_raw

    im_std = np.std(median)
    print("Standard deviation of the image is {:.2f}".format(im_std))

    max_pixel_crop = np.max(median[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area reaches {}".format(max_pixel_crop))
    if savefits_name is None:
        savefits_name = rawfile[:-4] + ".fits"
    elif savefits_name[-4:] != ("fits" or "fit"):
        print("Please provide a filename with fits or fit extension")
        exit(0)

    im2fits(median, savefits_name, overwrite=overwrite_fits)

    sew = sewpy.SEW(params=sewpy_params,
                    # config={"DETECT_MINAREA": DETECT_MINAREA, "BACK_SIZE": 128, "BACK_FILTERSIZE": 3,
                    config={"DETECT_MINAREA": DETECT_MINAREA, "BACK_SIZE": BACK_SIZE, "BACK_FILTERSIZE": 3,
                            "DETECT_THRESH": THRESH, "ANALYSIS_THRESH": THRESH, "DEBLEND_MINCONT": DEBLEND_MINCONT, })

    sew_out = sew(savefits_name)
    sew_out['table'].sort('FLUX_ISO')
    sew_out['table'].reverse()
    sew_out['table']['FLUX_AREA'] = (1. / 4.) * np.pi * sew_out['table']['KRON_RADIUS'] * sew_out['table'][
        'KRON_RADIUS'] * sew_out['table']['A_IMAGE'] * sew_out['table']['B_IMAGE']
    if clean:
        sew_out['table'] = sew_out['table'][sew_out['table'][
                                                'FLAGS'] <= 16]  # sew_out['table'] = sew_out['table'][(sew_out['table']['FLUX_ISO'] / sew_out['table']['FLUX_AREA']) > 0.3]

    ymin = 0
    ymax = 0
    if search_xs[1] > 0 and search_ys[1] >= 0 and search_ys[0] != search_ys[1]:
        if search_ys[1] < search_ys[0]:
            ymin = search_ys[1]
            ymax = search_ys[0]
        else:
            ymin = search_ys[0]
            ymax = search_ys[1]
        print("Only searching in X {} to {} and Y {} to {}".format(search_xs[0], search_xs[1], ymin, ymax))
        sew_out['table'] = sew_out['table'][
            (sew_out['table']['X_IMAGE'] <= search_xs[1]) & (sew_out['table']['X_IMAGE'] >= search_xs[0]) & (
                    sew_out['table']['Y_IMAGE'] <= ymax) & (sew_out['table']['Y_IMAGE'] >= ymin)]

    n_sources = len(sew_out['table'])
    ID_ = Column(range(n_sources), name='ID')
    sew_out['table'].add_column(ID_, index=0)
    sew_out['table'].add_index('ID')

    print("Found {} sources in file {}".format(n_sources, rawfile))

    plot_sew_cat(median, sew_out, outfile=saveplot_name, xlim=cropxs, ylim=cropys, vmax=max_pixel_crop,
                 pattern_label_x_min=search_xs[0], pattern_label_x_max=search_xs[1], pattern_label_y_min=ymin,
                 pattern_label_y_max=ymax, df_LEDs=df_LEDs, show=show)
    if savecatalog_name is not None:
        from astropy.io import ascii
        ascii.write(sew_out['table'], savecatalog_name, overwrite=True)
    if show:
        plt.show(block=False)
    return sew_out['table'], median


def plot_raw_cat(rawfile, sewtable, df=None, center_pattern=np.array([1891.25, 1063.75]), cropxs=(1050, 2592),
                 cropys=(1850, 250), kernel_w=3, save_catlog_name="temp_ring_search_cat.txt", df_LEDs=None,
                 center_offset=[0, 0], save_for_vvv="temp_ring_vvv_XY_pix.csv", saveplot_name=None, show=False,
                 verbose=False):
    '''
    Plots raw file (path to .RAW image) with imshow. If there is a 'df' object - this object is assumed to be the VVV
    list of panels in the ring - find minimum distance between that source with source in the sewtable and assign that
    panel number to the sewtable object. Plot the resulting image.

    '''
    from astropy.table import Column
    im_raw = read_image(rawfile)
    if has_cv2:
        median = cv2.medianBlur(im_raw, kernel_w)
    else:
        print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
        median = im_raw

    max_pixel_crop = np.max(median[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])

    n_sources = len(sewtable)
    ID_ = Column(range(n_sources), name='Panel_ID_guess')
    sewtable.add_column(ID_, index=0)
    sewtable.add_index('Panel_ID_guess')
    vvvID_ = Column(range(n_sources), name='#')
    sewtable.add_column(vvvID_, index=0)
    sewtable.add_index('#')

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    ax_img = ax.imshow(median, vmax=max_pixel_crop, cmap='gray')
    # else:
    # ax_img = ax.imshow(dst_trans, cmap='gray')
    fig.colorbar(ax_img)

    # plt.plot(cat_test['X_IMAGE'], cat_test['Y_IMAGE'], color='r',  marker='o', ms=5, ls='')
    # plt.scatter(sew_out_trans['table']['X_IMAGE'],
    #            sew_out_trans['table']['Y_IMAGE'], s=40, facecolors='none', edgecolors='r')

    i = 0

    for row in sewtable:
        i += 1
        e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2, height=row['B_IMAGE'] * 2,
                    angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('c')
        ax.add_artist(e)

        if df is not None:
            df.loc[:, 'tmpdist2'] = (df['Xpix'] - row['X_IMAGE']) ** 2 + (df['Ypix'] - row['Y_IMAGE']) ** 2
            pID = df.loc[df['tmpdist2'].idxmin(), 'Panel']
            vvvID = df.loc[df['tmpdist2'].idxmin(), '#']
            print("Guess this is panel {}".format(pID))
            if i > 1:
                # if pID == sewtable['Panel_ID_guess'][i-1] or vvvID:
                if pID in sewtable['Panel_ID_guess']:
                    print("Panel {} already found...".format(pID))
                    print("Sorry I haven't figured out this part yet")
                    """
                    print("Let's try a phase check")
                    df.loc[:, 'tmpistphase'] = abs(df['Phase'] - row['Phase'])
                    pID = df.loc[df['tmpistphase'].idxmin(), 'Panel']
                    vvvID = df.loc[df['tmpistphase'].idxmin(), '#']
                    print("Now I guess this is panel {}".format(pID))
                    """
            sewtable['Panel_ID_guess'][i - 1] = pID
            sewtable['#'][i - 1] = vvvID
            plt.plot(df['Xpix'], df['Ypix'], 'm.', markersize=4, alpha=0.3)
        else:
            pID = int(row['ID'])
        xytext_ = np.array(
            [row['X_IMAGE'] + row['X_IMAGE'] - center_pattern[0], row['Y_IMAGE'] + row['Y_IMAGE'] - center_pattern[1]])
        xytext_[0] = min(xytext_[0], 2700)
        xytext_[0] = max(xytext_[0], 800)
        xytext_[1] = min(xytext_[1], 2000)
        xytext_[1] = max(xytext_[1], 200)
        ax.annotate(pID, xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data',
                    # xytext=(np.array([row['X_IMAGE'] - 40, row['Y_IMAGE'] - 40])), # for orig lens
                    xytext=(xytext_),  # for orig lens
                    # xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                    color='c', alpha=0.8,
                    arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                    alpha=0.7), )

    if df_LEDs is not None:
        for i, row in df_LEDs.iterrows():
            e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2,
                        height=row['B_IMAGE'] * 2, angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('gold')
            ax.add_artist(e)
        center = [np.mean(df_LEDs['X_IMAGE']) + center_offset[0], np.mean(df_LEDs['Y_IMAGE']) + center_offset[1]]
        plt.plot([center[0]], [center[1]], color='gold', marker='+')

        # plot the CM after perspective transform
        LED_coords = df_LEDs[['X_IMAGE', 'Y_IMAGE']].to_numpy(dtype='float32')
        #print(LED_coords.shape, LED_REF.shape)
        if LED_coords.shape[0] == 4 or LED_coords.shape[0] == 8:
            cm = get_CM_coords(LED_coords)
            for i, row in cm:
                cm_center = np.mean(cm, axis=0)
                ax.plot([cm_center[0]], [cm_center[1]], 'g+', alpha=0.4)
                ax.scatter(cm[:, 0], cm[:, 1], s=2, facecolors='none', edgecolors='g', alpha=0.4)

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    plt.tight_layout()

    if saveplot_name is not None:
        fig.savefig(saveplot_name)
    if show:
        plt.show(block=False)

    from astropy.io import ascii
    ascii.write(sewtable.group_by('#'), save_catlog_name, overwrite=True)

    if df is not None:
        # no 'crappy' files for vvv (N.B 'crappy' is undefined and subject to change)
        df_vvv = sewtable.to_pandas()
        # df_vvv = df_vvv[
        #    ['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE', "A_x_KR_in_pix", "B_x_KR_in_pix", "THETA_IMAGE", 'FLUX_AREA',
        #     'KRON_RADIUS']]
        df_vvv = df_vvv[VVV_COLS]
        df_vvv = df_vvv.sort_values('#').reset_index(drop=True)
        cxs = []
        cys = []
        m00s = []
        m11s = []
        m12s = []
        m20s = []
        m21s = []
        m22s = []
        m31s = []
        m32s = []
        m33s = []
        m34s = []
        m40s = []
        m41s = []
        m42s = []
        m43s = []
        m44s = []
        for i in range(len(df_vvv)):
            # skx, sky = get_skewness(median, df_vvv, pind=i, show=False, verbose=True, reshape=True)
            stats_dict = get_skewness(median, df_vvv, r_ellipse=2, pind=i, show=(show and verbose), verbose=verbose, reshape=True)
            cxs.append(stats_dict['cx'])
            cys.append(stats_dict['cy'])
            m00s.append(stats_dict['m00'])
            m11s.append(stats_dict['m11'])
            m12s.append(stats_dict['m12'])
            m20s.append(stats_dict['m20'])
            m21s.append(stats_dict['m21'])
            m22s.append(stats_dict['m22'])
            m31s.append(stats_dict['m31'])
            m32s.append(stats_dict['m32'])
            m33s.append(stats_dict['m33'])
            m34s.append(stats_dict['m34'])
            m40s.append(stats_dict['m40'])
            m41s.append(stats_dict['m41'])
            m42s.append(stats_dict['m42'])
            m43s.append(stats_dict['m43'])
            m44s.append(stats_dict['m44'])
        df_vvv['cx'] = cxs
        df_vvv['cy'] = cys
        df_vvv['m00'] = m00s
        df_vvv['m11'] = m11s
        df_vvv['m12'] = m12s
        df_vvv['m20'] = m20s
        df_vvv['m21'] = m21s
        df_vvv['m22'] = m22s
        df_vvv['m31'] = m31s
        df_vvv['m32'] = m32s
        df_vvv['m33'] = m33s
        df_vvv['m34'] = m34s
        df_vvv['m40'] = m40s
        df_vvv['m41'] = m41s
        df_vvv['m42'] = m42s
        df_vvv['m43'] = m43s
        df_vvv['m44'] = m44s

        print("Mean center X {} Y {}".format(np.mean(df_vvv['X_IMAGE']), np.mean(df_vvv['Y_IMAGE'])))
        print("Sum tails vector: (M31, M32) = {:.2f}, {:.2f} ".format(np.sum(df_vvv['m31']), np.sum(df_vvv['m32'])))

        df_vvv.to_csv(save_for_vvv, index=False)

    return


def quick_check_raw_ring(rawfile, save_for_vvv="temp_ring_vvv_XY_pix.csv", saveplot_name=None, show=False, kernel_w=3,
                         cropxs=(1050, 2592), cropys=(1850, 250), labelcolor='steelblue', labelalpha=0.3, textalpha=0.9,
                         plot_center=True, df_LEDs=None, center_offset=[0, 0], vmax=None):
    im_raw = read_image(rawfile)
    if has_cv2:
        median = cv2.medianBlur(im_raw, kernel_w)
    else:
        print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
        median = im_raw

    max_pixel_crop = np.max(median[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])

    df = pd.read_csv(save_for_vvv)
    n_sources = len(df)
    print("{} centroids found".format(n_sources))

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    if vmax is None:
        vmax = max_pixel_crop

    ax_img = ax.imshow(median, vmax=vmax, cmap='gray')
    # else:
    # ax_img = ax.imshow(dst_trans, cmap='gray')
    # fig.colorbar(ax_img)

    # plt.plot(cat_test['X_IMAGE'], cat_test['Y_IMAGE'], color='r',  marker='o', ms=5, ls='')
    # plt.scatter(sew_out_trans['table']['X_IMAGE'],
    #            sew_out_trans['table']['Y_IMAGE'], s=40, facecolors='none', edgecolors='r')

    # plt.plot(df['X_IMAGE'], df['Y_IMAGE'], 'c.', markersize=4, alpha=0.3)
    center_pattern = [np.mean(df['X_IMAGE']), np.mean(df['Y_IMAGE'])]

    for i, row in df.iterrows():
        e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2, height=row['B_IMAGE'] * 2,
                    angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=labelalpha)
        e.set_clip_box(ax.bbox)
        e.set_alpha(labelalpha)
        e.set_color(labelcolor)
        ax.add_artist(e)

        xytext_ = np.array(
            [row['X_IMAGE'] + row['X_IMAGE'] - center_pattern[0], row['Y_IMAGE'] + row['Y_IMAGE'] - center_pattern[1]])
        xytext_[0] = min(xytext_[0], 2700)
        xytext_[0] = max(xytext_[0], 800)
        xytext_[1] = min(xytext_[1], 2000)
        xytext_[1] = max(xytext_[1], 200)
        ax.annotate(int(row['Panel_ID_guess']), xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data',
                    #            # xytext=(np.array([row['X_IMAGE'] - 40, row['Y_IMAGE'] - 40])), # for orig lens
                    #            xytext=(np.array([row['X_IMAGE'] + (row['X_IMAGE'] - center_pattern[0]) * 1.3,
                    #                              row['Y_IMAGE'] + (row['Y_IMAGE'] - center_pattern[1]) * 1.3])),  # for orig lens
                    # xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                    xytext=(xytext_),  # for orig lens
                    color=labelcolor, alpha=textalpha,
                    arrowprops=dict(facecolor=labelcolor, edgecolor=labelcolor, shrink=0.05, headwidth=0.5,
                                    headlength=4, width=0.2, alpha=labelalpha), )

    if df_LEDs is not None:
        for i, row in df_LEDs.iterrows():
            e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2,
                        height=row['B_IMAGE'] * 2, angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('gold')
            ax.add_artist(e)
        center = [np.mean(df_LEDs['X_IMAGE']) + center_offset[0], np.mean(df_LEDs['Y_IMAGE']) + center_offset[1]]
        plt.plot([center[0]], [center[1]], color='gold', marker='+')

        # plot the CM after perspective transform
        LED_coords = df_LEDs[['X_IMAGE', 'Y_IMAGE']].to_numpy(dtype='float32')
        if LED_coords.shape[0] == 4 or LED_coords.shape[0] == 8:
            cm = get_CM_coords(LED_coords)
            for i, row in cm:
                cm_center = np.mean(cm, axis=0)
                ax.plot([cm_center[0]], [cm_center[1]], 'g+', alpha=0.4)
                ax.scatter(cm[:, 0], cm[:, 1], s=2, facecolors='none', edgecolors='g', alpha=0.4)

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)
    if plot_center:
        plt.plot([center_pattern[0]], [center_pattern[1]], color='c', marker='x')
    if saveplot_name is not None:
        print("saving to file {}".format(saveplot_name))
        plt.savefig(saveplot_name)
    if show:
        plt.show()


def naive_comparison(sew_out_table1, sew_out_table2, im1, im2, min_dist=20, outfile1=None, outfile2=None, verbose=False,
                     diffcat1="diff_cat1.txt", diffcat2="diff_cat2.txt", cropxs=(1050, 2592), cropys=(1410, 610),
                     center=np.array([1891.25, 1063.75])):
    x_corners, y_corners = get_central_mod_corners(center=center)

    # xy_common=[]
    diff_ind1 = range(len(sew_out_table1))
    diff_ind2 = range(len(sew_out_table2))
    commond_ind1 = []
    commond_ind2 = []

    with open(diffcat1, 'w') as diffcat1_io:
        diffcat1_io.write(" ".join(sew_out_table1.colnames))
        diffcat1_io.write("\n")
    with open(diffcat2, 'w') as diffcat2_io:
        diffcat2_io.write(" ".join(sew_out_table2.colnames))
        diffcat2_io.write("\n")

    max_pixel_crop2 = np.max(im2[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    # print("Brightest pixel in the zoomed area in image 2 reaches {}".format(max_pixel_crop2))
    # if max_pixel_crop2 == 255:
    #    print("Image 2 is saturated. ")

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax_img = ax.imshow(im2, cmap='gray', vmax=max_pixel_crop2)
    fig.colorbar(ax_img)

    i2 = -1

    for row in sew_out_table2:
        x2_ = row['X_IMAGE']
        y2_ = row['Y_IMAGE']
        f2_ = row['FLUX_ISO']
        xy2_ = np.array([x2_, y2_])
        i2 += 1
        i1 = -1
        for row1 in sew_out_table1:
            i1 += 1
            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            xy1_ = np.array([x1_, y1_])
            dist_ = np.linalg.norm(xy1_ - xy2_)
            if dist_ <= min_dist:
                # xy_common.append((xy1_+xy2_)/2.)
                commond_ind1.append(i1)
                commond_ind2.append(i2)
                e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), width=row['A_IMAGE'] * 2,
                            height=row['B_IMAGE'] * 2, angle=row['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('r')

                # if row['X_IMAGE'] <= 2100 and row['X_IMAGE'] >= 1700 and row['Y_IMAGE'] >= 880 and row[
                #    'Y_IMAGE'] <= 1250:
                if row['X_IMAGE'] <= PATTERN_LABEL_X_MAX and row['X_IMAGE'] >= PATTERN_LABEL_X_MIN and row[
                    'Y_IMAGE'] >= PATTERN_LABEL_Y_MIN and row['Y_IMAGE'] <= PATTERN_LABEL_Y_MAX:
                    # print("Yo")
                    # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                    ax.annotate(int(row['ID']), xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data',
                                xytext=(np.array([row['X_IMAGE'] + row['X_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[0],
                                                  row['Y_IMAGE'] + row['Y_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[
                                                      1]])),  # for orig lens
                                # xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                                color='c', alpha=0.8,
                                arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                                width=0.5, alpha=0.7), )

                # e.set_facecolor('r')
                try:
                    diff_ind1.remove(i1)
                except:
                    # if verbose:
                    #    print("Double Match 1...... !!!!!")
                    #    print(diff_ind1)
                    # else:
                    pass
                try:
                    diff_ind2.remove(i2)
                except:
                    # if verbose:
                    #    print("Double Match 2...... !!!!!")
                    #    print(diff_ind2)
                    # else:
                    pass

    i2 = -1
    for row1 in sew_out_table2:
        i2 += 1
        if i2 not in diff_ind2:
            continue
        x1_ = row1['X_IMAGE']
        y1_ = row1['Y_IMAGE']
        f1_ = row1['FLUX_ISO']
        xy1_ = np.array([x1_, y1_])
        if verbose:
            print("==== New in catalog 2 ====")
            print(xy1_)

        with open(diffcat2, 'a') as diffcat2_io:
            for c_ in sew_out_table2.colnames:
                diffcat2_io.write(str(row1[c_]))
                diffcat2_io.write(" ")
            diffcat2_io.write("\n")

        # xy_diff.append((xy1_+xy2_)/2.)

        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                    height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('y')

        # if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
        #    'Y_IMAGE'] <= 1250:
        if row1['X_IMAGE'] <= PATTERN_LABEL_X_MAX and row1['X_IMAGE'] >= PATTERN_LABEL_X_MIN and row1[
            'Y_IMAGE'] >= PATTERN_LABEL_Y_MIN and row1['Y_IMAGE'] <= PATTERN_LABEL_Y_MAX:
            # print("Yo")
            # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8, xycoords='data',
                        xytext=(np.array([row1['X_IMAGE'] + row1['X_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[0],
                                          row1['Y_IMAGE'] + row1['Y_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[1]])),
                        # for orig lens
                        # xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                        color='c', alpha=0.8,
                        arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2, width=0.5,
                                        alpha=0.7), )

    # xy_common=np.array(xy_common)
    # xy_diff=np.array(xy_diff)
    # print(diff_ind1, diff_ind2)
    # xy_common.shape #, xy_diff.shape
    if verbose:
        print("Common indices")
        print(commond_ind1, commond_ind2)

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)
    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners, y_corners, s=20, facecolors='none', edgecolors='m')

    if outfile2 is not None:
        plt.savefig(outfile2)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    max_pixel_crop1 = np.max(im1[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    # print("Brightest pixel in the zoomed area in image 1 reaches {}".format(max_pixel_crop1))
    # if max_pixel_crop1 == 255:
    #    print("Image 1 is saturated. ")

    ax_img = ax.imshow(im1, cmap='gray', vmax=max_pixel_crop1)
    fig.colorbar(ax_img)

    i1 = -1
    for row1 in sew_out_table1:
        i1 += 1
        x1_ = row1['X_IMAGE']
        y1_ = row1['Y_IMAGE']
        f1_ = row1['FLUX_ISO']
        xy1_ = np.array([x1_, y1_])

        dist_ = np.linalg.norm(xy1_ - xy2_)
        if i1 in commond_ind1:
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                        height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('r')

            # if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
            #    'Y_IMAGE'] <= 1250:
            if row1['X_IMAGE'] <= PATTERN_LABEL_X_MAX and row1['X_IMAGE'] >= PATTERN_LABEL_X_MIN and row1[
                'Y_IMAGE'] >= PATTERN_LABEL_Y_MIN and row1['Y_IMAGE'] <= PATTERN_LABEL_Y_MAX:
                # print("Yo")
                # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8, xycoords='data',
                            xytext=(np.array([row1['X_IMAGE'] + row1['X_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[0],
                                              row1['Y_IMAGE'] + row1['Y_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[
                                                  1]])),  # for orig lens

                            # xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                            color='c', alpha=0.8,
                            arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                            width=0.5, alpha=0.7), )

    i1 = -1
    for row1 in sew_out_table1:
        i1 += 1
        if i1 not in diff_ind1:
            continue

        x1_ = row1['X_IMAGE']
        y1_ = row1['Y_IMAGE']
        f1_ = row1['FLUX_ISO']
        xy1_ = np.array([x1_, y1_])
        # xy_diff.append((xy1_+xy2_)/2.)
        if verbose:
            print("==== New in catalog 1 ====")
            print(xy1_)
        with open(diffcat1, 'a') as diffcat1_io:
            for c_ in sew_out_table1.colnames:
                diffcat1_io.write(str(row1[c_]))
                diffcat1_io.write(" ")
            diffcat1_io.write("\n")
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                    height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('g')
        # if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
        #    'Y_IMAGE'] <= 1250:
        if row1['X_IMAGE'] <= PATTERN_LABEL_X_MAX and row1['X_IMAGE'] >= PATTERN_LABEL_X_MIN and row1[
            'Y_IMAGE'] >= PATTERN_LABEL_Y_MIN and row1['Y_IMAGE'] <= PATTERN_LABEL_Y_MAX:
            # print("Yo")
            # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8, xycoords='data',
                        xytext=(np.array([row1['X_IMAGE'] + row1['X_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[0],
                                          row1['Y_IMAGE'] + row1['Y_IMAGE'] - PATTERN_CENTER_FROM_LABEL_BOUNDS[1]])),
                        # for orig lens
                        # xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                        color='c', alpha=0.8,
                        arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2, width=0.5,
                                        alpha=0.7), )

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    # mark center and corners:
    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners, y_corners, s=20, facecolors='none', edgecolors='m')

    if outfile1 is not None:
        plt.savefig(outfile1)


def calc_dist(cat1, cat2, n1, n2, pix2mm=0.48244):
    dx = (cat2.loc[n2]['X_IMAGE'] - cat1.loc[n1]['X_IMAGE'])
    dy = (cat2.loc[n2]['Y_IMAGE'] - cat1.loc[n1]['Y_IMAGE'])
    dist_pix = np.sqrt(dx ** 2 + dy ** 2)
    dist_mm = dist_pix * pix2mm
    print("centroid moved in x {:.4f} pix and in y {:.4f} pix".format(dx, dy))
    print("centroid moved by distance {:.4f} pix = {:.4f} mm".format(dist_pix, dist_mm))

    return dx, dy, dist_pix


def plot_diff_labelled(rawf1, rawf2, cat1, cat2, ind1=None, ind2=None, motion_outfile_prefix="motion_output",
                       outfile1="new_label1.pdf", outfile2="new_label2.pdf", cropxs=(1350, 1800), cropys=(1250, 800),
                       center=np.array([1891.25, 1063.75])):
    x_corners, y_corners = get_central_mod_corners(center=center)

    # cat1 and cat2 are the * diff * cats
    im1 = read_image(rawf1)
    im2 = read_image(rawf2)

    kernel_w = 3
    im1 = cv2.medianBlur(im1, kernel_w)
    im2 = cv2.medianBlur(im2, kernel_w)

    cat1 = pd.read_csv(cat1, sep=r"\s+")
    cat2 = pd.read_csv(cat2, sep=r"\s+")

    # cat1_screen = cat1[(cat1.X_IMAGE >= np.min(x_corners)) & (cat1.X_IMAGE <= np.max(x_corners))]
    # cat2_screen = cat1[(cat2.Y_IMAGE >= np.min(y_corners)) & (cat2.Y_IMAGE <= np.max(y_corners))]

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    max_pixel_crop1 = np.max(im1[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area in image 1 reaches {}".format(max_pixel_crop1))
    if max_pixel_crop1 == 255:
        sat_pix = np.sum(im1 == 255)
        percent_satur = 100.0 * sat_pix / (cropys[0] - cropys[1]) / (cropxs[1] - cropxs[0])
        print("{:.3f}% of the pixels (= {} pixels) in zoomed area of image 1 is saturated. ".format(percent_satur,
                                                                                                    sat_pix))

    ax_img = ax.imshow(im1, cmap='gray', vmax=max_pixel_crop1)
    # ax_img = ax.imshow(im1, cmap='gray')
    fig.colorbar(ax_img)

    if ind1 is not None:
        motion_outfile = motion_outfile_prefix + "ind" + str(ind1) + "_ind" + str(ind2) + ".txt"

        row1 = cat1.loc[ind1]
        with open(motion_outfile, 'w') as io_:
            io_.write("# center (hard coded): ({}, {})\n".format(center[0], center[1]))
            io_.write(" ".join(cat1.columns))
            io_.write("\n")
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                    height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('g')
        # ax.annotate(str(ind1), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
        ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                    # xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                    xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                    color='g', alpha=0.8,
                    arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                    alpha=0.7), )
    else:
        for i, row1 in cat1.iterrows():
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                        height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')
            # ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),size=8,
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                        # xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                        color='g', alpha=0.8,
                        arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                        alpha=0.7), )

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners, y_corners, s=20, facecolors='none', edgecolors='m')

    plt.tight_layout()
    # plt.xlim(1170, 1970)
    # plt.ylim(1410, 610)

    plt.savefig(outfile1)

    # plt.show()

    max_pixel_crop2 = np.max(im2[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area in image 2 reaches {}".format(max_pixel_crop2))
    if max_pixel_crop2 == 255:
        sat_pix = np.sum(im2 == 255)
        percent_satur = 100.0 * sat_pix / (cropys[0] - cropys[1]) / (cropxs[1] - cropxs[0])
        print("{:.3f}% of the pixels (= {} pixels) in zoomed area of image 2 is saturated. ".format(percent_satur,
                                                                                                    sat_pix))

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax_img = ax.imshow(im2, cmap='gray', vmax=max_pixel_crop2)

    fig.colorbar(ax_img)

    if ind2 is not None:
        motion_outfile = motion_outfile_prefix + "ind" + str(ind1) + "_ind" + str(ind2) + ".txt"
        row1 = cat2.loc[ind2]
        with open(motion_outfile, 'a') as io_:
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                    height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('y')
        # ax.annotate(str(ind2), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
        ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                    # xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                    xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                    color='y', alpha=0.8,
                    arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                    alpha=0.7), )
    else:
        for i, row1 in cat2.iterrows():
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), width=row1['A_IMAGE'] * 2,
                        height=row1['B_IMAGE'] * 2, angle=row1['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')
            # ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                        # xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                        color='y', alpha=0.8,
                        arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                        alpha=0.7), )

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners, y_corners, s=20, facecolors='none', edgecolors='m')

    plt.tight_layout()
    # plt.xlim(1170, 1970)
    # plt.ylim(1410, 610)

    plt.savefig(outfile2)
    # plt.show()

    if ind1 is not None and ind2 is not None:
        dx, dy, dist_pix = calc_dist(cat1, cat2, ind1, ind2)
        with open(motion_outfile, 'a') as io_:
            io_.write("# dx = {:.4f} pix, dy = {:.4f} pix, distance = {:.4f} pix \n".format(dx, dy, dist_pix))


def get_datetime_rawname(raw_name):
    pattern = r'\b\w{1,4}-\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2}'
    match = re.search(pattern, raw_name)
    dt_match = match.group()  # raw_name[:match.start()]
    dt_match = "_".join(dt_match.split('-'))
    dt_match = "_".join(dt_match.split(':'))
    return dt_match


def yes_or_no(question):
    # from builtins import input
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter y or n")


def grid2gif(im1, im2, output_gif):
    str1 = 'convert -delay 50 -loop 0 -quality 100 -density 144 ' + im1 + ' ' + im2 + ' ' + output_gif
    subprocess.call(str1, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Compare two raw images of stars at the focal plane')

    parser.add_argument('rawfile1', type=str)
    parser.add_argument('rawfile2', type=str, default=None, nargs='?')

    parser.add_argument('--DETECT_MINAREA', type=int, default=30, help="+++ Important parameter +++: "
                                                                       "Config param for sextractor, our default is 30.")
    parser.add_argument('--DETECT_MINAREA_S2', type=int, default=500, help="+++ Important parameter +++: "
                                                                           "Config param for sextractor for P2S2 (outer ring) images, our default is 500.")
    parser.add_argument('--THRESH', type=int, default=6, help="+++ Important parameter +++: "
                                                              "Config param for sextractor, our default is 6.")
    parser.add_argument('--DEBLEND_MINCONT', type=float, default=0.01, help="+++ Important parameter +++: "
                                                                            "Config param for sextractor, our default is 0.01 "
                                                                            "The smaller this number is, the harder we try to "
                                                                            "deblend, i.e. to separate overlaying objects. ")
    parser.add_argument('--BACK_SIZE', type=int, default=128, help="+++ Important parameter +++: "
                                                                   "Config param for sextractor, our default is 128."
                                                                   "Change to smaller values if you need to estimate very localized background."
                                                                   "32 can be used for cases with bright reflection from the moon. ")
    parser.add_argument('--kernel_w', type=int, default=3,
                        help="If you have cv2, this is the median blurring kernel width"
                             "our default is 3 (for a 3x3 kernel).")

    parser.add_argument('--min_dist', type=float, default=20, help="+++ Important parameter +++: "
                                                                   "Minimum distance we use to conclude a centroid is common in both images"
                                                                   "our default is 20 pixels.")

    parser.add_argument('--save_filename_prefix1', default=None, help="File name prefix of the output files, "
                                                                      "this will automatically populate savefits_name1, saveplot_name1, savecatalog_name1, and diffcatalog_name1"
                                                                      "default is None.")

    parser.add_argument('--save_filename_prefix2', default=None, help="File name prefix of the output files, "
                                                                      "this will automatically populate savefits_name2, saveplot_name2, savecatalog_name2, and diffcatalog_name2"
                                                                      "default is None.")

    parser.add_argument('--savefits_name1', default=None,
                        help="File name of the fits for the first image if you want to choose, "
                             "default has the same name as raw.")
    parser.add_argument('--saveplot_name1', default=None,
                        help="File name of the image (jpeg or pdf etc) for the first image if you want to choose, "
                             "default is not to save it.")
    parser.add_argument('--savefits_name2', default=None,
                        help="File name of the fits for the second image if you want to choose, "
                             "default has the same name as raw.")
    parser.add_argument('--saveplot_name2', default=None,
                        help="File name of the image (jpeg or pdf etc) for the second image if you want to choose, "
                             "default is not to save it.")
    parser.add_argument('--savecatalog_name1', default=None,
                        help="File name of the ascii catalog derived from the first image, "
                             "default is not to save it.")
    parser.add_argument('--savecatalog_name2', default=None,
                        help="File name of the ascii catalog derived from the second image, "
                             "default is not to save it.")
    parser.add_argument('--diffcatalog_name1', default="diff_cat1.txt",
                        help="File name of the ascii catalog for sources only in the first image, "
                             "default is diff_cat1.txt.")
    parser.add_argument('--diffcatalog_name2', default="diff_cat2.txt",
                        help="File name of the ascii catalog for sources only in the second image, "
                             "default is diff_cat2.txt.")
    parser.add_argument('--diffplot_name1', default="diff_cat1.pdf",
                        help="File name of the image catalog for sources only in the first image, "
                             "default is diff_cat1.pdf.")
    parser.add_argument('--diffplot_name2', default="diff_cat2.pdf",
                        help="File name of the image catalog for sources only in the second image, "
                             "default is diff_cat2.pdf.")
    parser.add_argument('--datadir', default="data",
                        help="Folder to save all output files. Default is ./data (ignored by git)")
    parser.add_argument('--gifname', default=None,  # default="compare.gif",
                        help="File name to save gif animation. ")

    parser.add_argument('--cropx1', default=580, type=int,  # default=1170,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")
    parser.add_argument('--cropx2', default=2592, type=int,  # default=1970,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")

    parser.add_argument('--cropy1',  # default=1410,
                        default=1944, type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")
    parser.add_argument('--cropy2',  # default=610,
                        default=150, type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")

    parser.add_argument('-o', '--motion_outfile_prefix', dest="motion_outfile_prefix", default="motion_output",
                        help="File name prefix of the output catalog to calculate motion.")

    parser.add_argument('--nozoom', action='store_true', help="Do not zoom/crop the image. ")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--clean', action='store_true', help="Whether or not to delete centroid with flag > 16.")
    parser.add_argument('-s', '--single', action='store_true', help="Only analyze a single image.")
    parser.add_argument('-r', '--ring', action='store_true', help="Try to find a ring.")
    parser.add_argument('--p1rx', default=0, type=float,
                        help="This is just for S1 alignment, P1 rx applied to check for ghost images due to S1 misalignment. Only a few values are valid. ")
    parser.add_argument('--p1ry', default=0, type=float,
                        help="This is just for P1 alignment, P1 ry applied to check for tube dragging by overshooting center. Only a few values are valid. ")
    parser.add_argument('--p2ry', default=0, type=float,
                        help="This is just for P2 alignment, P2 ry applied to check for tube dragging by overshooting center. Only a few values are valid. ")

    parser.add_argument('--clustering', action='store_true')

    parser.add_argument('-C', '--center', nargs=2, type=float, default=[1891.25, 1063.75],
                        help="Center coordinate X_pix Y_pix. ")
    parser.add_argument('-p', '--pattern_center', nargs=2, type=float, default=None,  # default=[1891.25, 1063.75],
                        help="Center coordinate for ring pattern X_pix Y_pix. ")
    parser.add_argument('--vvv_tag', default=None, help="A string to identify which ring. ")

    parser.add_argument('--ring_rad', type=float, default=32 / PIX2MM, help="Radius in pixels for the inner ring pattern. ")
    parser.add_argument('--P2S1ring_rad', type=float, default=32 * 1.59 / PIX2MM, help="Radius in pixels for the middle P2 S1 ring pattern. ")
    parser.add_argument('--P2S2ring_rad', type=float, default=32 * 2.2 / PIX2MM, help="Radius in pixels for the outermost P2 S2 ring pattern. ")
    parser.add_argument('--ring_tol', type=float, default=0.1)
    parser.add_argument('--phase_offset_rad', type=float, default=0.)
    parser.add_argument('--ring_frac', type=float, default=0.2,
                        help="Fraction (1-frac, 1+frac)*radius that you accept a centroid "
                             "as part of a ring pattern. ")

    parser.add_argument('--ring_file', default=None, help="File name for ring pattern. ")
    parser.add_argument('--labelcolor', default='c', help="Label color. ")
    parser.add_argument('--search_xs', nargs=2, type=float, default=[0, 0],
                        # parser.add_argument('--search_xs', nargs=2, type=float, default=[1050, 2592],
                        help="Xmin and Xmax to list all centroid in a box. ")
    parser.add_argument('--search_ys', nargs=2, type=float, default=[0, 0],
                        # parser.add_argument('--search_ys', nargs=2, type=float, default=[1850, 250],
                        help="Ymin and Ymax to list all centroid in a box. ")
    parser.add_argument('--quick_ring_check', default=None, help="Do ring check; dubs as file name for ring pattern. ")
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--get_center", action='store_true')
    parser.add_argument("--skip_p2", action='store_true')
    parser.add_argument("--skip_s2", action='store_true')
    parser.add_argument("--psf", action='store_true')
    parser.add_argument("--psf_search_width", type=float, default=50)
    # parser.add_argument("--LED_search_width", type=float, default=800 )
    parser.add_argument('--LED_search_xs', nargs=2, type=float, default=[630, 2560],
                        help="Xmin and Xmax to search for LED centroid in a box. ")
    parser.add_argument('--LED_search_ys', nargs=2, type=float, default=[200, 1860],
                        help="Ymin and Ymax to search for LED centroid in a box. ")
    parser.add_argument("--ring_vmax", type=float, default=None)
    parser.add_argument("--inner8", action='store_true')


    args = parser.parse_args()

    if args.nozoom:
        cropxs = None
        cropys = None
    else:
        cropxs = (args.cropx1, args.cropx2)
        cropys = (args.cropy1, args.cropy2)

    if not os.path.exists(args.datadir):
        sure = yes_or_no("Are you sure to make a new directory {} to save data? ".format(args.datadir))
        if sure:
            os.mkdir(args.datadir)
        else:
            print("Okay, mission abort.")
            exit(0)

    ring_file = None

    if args.save_filename_prefix1 is not None:
        savefits_name1 = os.path.join(args.datadir, args.save_filename_prefix1 + '_im1.fits')
        savecatalog_name1 = os.path.join(args.datadir, args.save_filename_prefix1 + '_cat1.txt')
        saveplot_name1 = os.path.join(args.datadir, args.save_filename_prefix1 + '_cat1.pdf')
        diffcatalog_name1 = os.path.join(args.datadir, args.save_filename_prefix1 + "_diff_cat1.txt")
        diffplot_name1 = os.path.join(args.datadir, args.save_filename_prefix1 + "_diff_cat1.pdf")
        motion_outfile_prefix = os.path.join(args.datadir, args.save_filename_prefix1)
        save_filename_prefix1 = args.save_filename_prefix1
        if args.ring_file is None:
            ring_file = save_filename_prefix1 + "_ring_search_1.pdf"
            ring_cat_file = save_filename_prefix1 + "_ring_search_1.txt"
        else:
            ring_file = os.path.join(args.datadir, args.ring_file)
            ring_cat_file = os.path.join(args.datadir, args.ring_file[:-4] + ".txt")
        vvv_ring_file = save_filename_prefix1 + "_ring_search_vvv.csv"

    elif args.savefits_name1 is None or args.savecatalog_name1 is None or args.diffcatalog_name1 is None or args.diffplot_name1 is None:
        dt_match = get_datetime_rawname(args.rawfile1)
        print("Using default output file names with date {}".format(dt_match))
        if dt_match < '2021_09_19_20_00_00':
            print("Before 2021 Sep 19; using old LED_REF coordinates")
            LED_REF = LED_REF_before2021Sep19
        save_filename_prefix1 = os.path.join(args.datadir, "res_focal_plane_" + dt_match)
        savefits_name1 = save_filename_prefix1 + '_im1.fits'
        savecatalog_name1 = save_filename_prefix1 + '_cat1.txt'
        saveplot_name1 = save_filename_prefix1 + '_cat1.pdf'
        diffcatalog_name1 = save_filename_prefix1 + "_diff_cat1.txt"
        diffplot_name1 = save_filename_prefix1 + "_diff_cat1.pdf"
        motion_outfile_prefix = save_filename_prefix1
        if args.ring_file is None:
            ring_file = save_filename_prefix1 + "_ring_search_1.pdf"
            ring_cat_file = save_filename_prefix1 + "_ring_search_1.txt"
        else:
            ring_file = os.path.join(args.datadir, args.ring_file)
            ring_cat_file = os.path.join(args.datadir, args.ring_file[:-4] + ".txt")
        vvv_ring_file = save_filename_prefix1 + "_ring_search_vvv.csv"
    else:
        savefits_name1 = os.path.join(args.datadir, args.savefits_name1)
        savecatalog_name1 = os.path.join(args.datadir, args.savecatalog_name1)
        saveplot_name1 = os.path.join(args.datadir, args.saveplot_name1)
        diffcatalog_name1 = os.path.join(args.datadir, args.diffcatalog_name1)
        diffplot_name1 = os.path.join(args.datadir, args.diffplot_name1)
        motion_outfile_prefix = os.path.join(args.datadir, args.motion_outfile_prefix)
    if args.rawfile2 is not None:
        if args.save_filename_prefix2 is not None:
            savefits_name2 = os.path.join(args.datadir, args.save_filename_prefix2 + '_im2.fits')
            savecatalog_name2 = os.path.join(args.datadir, args.save_filename_prefix2 + '_cat2.txt')
            saveplot_name2 = os.path.join(args.datadir, args.save_filename_prefix2 + '_cat2.pdf')
            diffcatalog_name2 = os.path.join(args.datadir, args.save_filename_prefix2 + "_diff_cat2.txt")
            diffplot_name2 = os.path.join(args.datadir, args.save_filename_prefix2 + "_diff_cat2.pdf")
            motion_outfile_prefix = motion_outfile_prefix + "_" + args.save_filename_prefix2 + "_motion.txt"
            gifname = motion_outfile_prefix + "_" + args.save_filename_prefix2 + "_anime.gif"
        elif args.savefits_name2 is None or args.savecatalog_name2 is None or args.diffcatalog_name2 is None or args.diffplot_name2 is None:
            dt_match = get_datetime_rawname(args.rawfile2)
            print("Using default output file names with date {}".format(dt_match))
            save_filename_prefix2 = os.path.join(args.datadir, "res_focal_plane_" + dt_match)
            savefits_name2 = save_filename_prefix2 + '_im2.fits'
            savecatalog_name2 = save_filename_prefix2 + '_cat2.txt'
            saveplot_name2 = save_filename_prefix2 + '_cat2.pdf'
            diffcatalog_name2 = save_filename_prefix2 + "_diff_cat2.txt"
            diffplot_name2 = save_filename_prefix2 + "_diff_cat2.pdf"
            gifname = motion_outfile_prefix + "_" + dt_match + "_anime.gif"
            motion_outfile_prefix = motion_outfile_prefix + "_" + dt_match + "motion"
        else:
            savefits_name2 = os.path.join(args.datadir, args.savefits_name2)
            savecatalog_name2 = os.path.join(args.datadir, args.savecatalog_name2)
            saveplot_name2 = os.path.join(args.datadir, args.saveplot_name2)
            diffcatalog_name2 = os.path.join(args.datadir, args.diffcatalog_name2)
            diffplot_name2 = os.path.join(args.datadir, args.diffplot_name2)
            motion_outfile_prefix = args.motion_outfile_prefix
            gifname = os.path.join(args.datadir, args.gifname)
        if args.gifname is not None:
            gifname = os.path.join(args.datadir, args.gifname)

    sew_params = SEWPY_PARAMS

    if (args.single or args.rawfile2 is None) and args.quick_ring_check is None:
        if args.psf:
            if args.pattern_center is None:
                xc, yc = get_centroid_global(sew_out_table1)
            else:
                xc, yc = args.pattern_center[0], args.pattern_center[1]
            # LED_search_width = args.LED_search_width
            # search_LEDxs = [xc-LED_search_width, xc+LED_search_width]
            # search_LEDys = [yc-LED_search_width, yc+LED_search_width]
            search_LEDxs = args.LED_search_xs
            search_LEDys = args.LED_search_ys
            search_xs = [xc - args.psf_search_width, xc + args.psf_search_width]
            search_ys = [yc - args.psf_search_width, yc + args.psf_search_width]
            sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                                  DETECT_MINAREA=args.DETECT_MINAREA, BACK_SIZE=args.BACK_SIZE,
                                                  THRESH=args.THRESH, DEBLEND_MINCONT=args.DEBLEND_MINCONT,
                                                  sewpy_params=sew_params, cropxs=cropxs, cropys=cropys,
                                                  clean=args.clean, savefits_name=savefits_name1, overwrite_fits=True,
                                                  saveplot_name=saveplot_name1, savecatalog_name=savecatalog_name1,
                                                  search_xs=search_LEDxs, search_ys=search_LEDys,
                                                  show=(args.show and not args.ring))
            print("Processing single image for LEDs. Done.")
            df_LEDs, center_LEDs = find_LEDs(sew_out_table1)
            if len(df_LEDs) == 4 or len(df_LEDs) == 8:  # hard coded for now; when the 8 LEDs are used, many more changes are needed for find_LEDs
                LED_filename = save_filename_prefix1 + "_LEDs.csv"
                df_LEDs.to_csv(LED_filename, index=False)
            sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                                  DETECT_MINAREA=args.DETECT_MINAREA, BACK_SIZE=args.BACK_SIZE,
                                                  THRESH=args.THRESH, DEBLEND_MINCONT=args.DEBLEND_MINCONT,
                                                  sewpy_params=sew_params, cropxs=cropxs, cropys=cropys,
                                                  clean=args.clean, savefits_name=savefits_name1, overwrite_fits=True,
                                                  saveplot_name=saveplot_name1, savecatalog_name=savecatalog_name1,
                                                  search_xs=search_xs, search_ys=search_ys,
                                                  show=(args.show and not args.ring), df_LEDs=df_LEDs)
            print("Processing single image. Done.")

            # Preparing a file for VVV
            df_centroid = pd.read_csv(savecatalog_name1, sep=r"\s+")
            df_centroid = df_centroid.append(pd.Series("", index=df_centroid.columns), ignore_index=True)
            df = pd.concat([df_centroid, df_LEDs])
            df.to_csv(save_filename_prefix1 + "_centerPSF_and_LEDs.csv", index=False)

        else:
            search_xs = args.search_xs
            search_ys = args.search_ys
            sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                                  DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                                                  DEBLEND_MINCONT=args.DEBLEND_MINCONT, BACK_SIZE=args.BACK_SIZE,
                                                  sewpy_params=sew_params, cropxs=cropxs, cropys=cropys,
                                                  clean=args.clean, savefits_name=savefits_name1, overwrite_fits=True,
                                                  saveplot_name=saveplot_name1, savecatalog_name=savecatalog_name1,
                                                  search_xs=search_xs, search_ys=search_ys,
                                                  show=(args.show and not args.ring))
            print("Processing single image. Done.")
            df_LEDs, center_LEDs = find_LEDs(sew_out_table1)
            if len(df_LEDs) == 4 or len(df_LEDs) == 8:  # hard coded for now; when the 8 LEDs are used, many more changes are needed for find_LEDs
                LED_filename = save_filename_prefix1 + "_LEDs.csv"
                df_LEDs.to_csv(LED_filename, index=False)
        chooseinner = False
        if args.ring:
            # new for S1 alignment
            if (args.p1rx == 0) and (args.p1ry == 0) and (args.p2ry == 0):
                all_panels = DEFAULT_CENTROID_LAYOUT
                num_vvv = NUM_VVV_DEFAULT
            elif args.p1rx == -1:
                print("Using Rx -1 centroid layout for S1 alignment. ")
                all_panels = RXm1_CENTROID_LAYOUT
                num_vvv = NUM_VVV_DEFAULT
            elif args.p1rx == -2 or args.p1rx == -3:
                print("Using Rx {} centroid layout for S1 alignment. ".format(args.p1rx))
                all_panels = RXm2_CENTROID_LAYOUT
                num_vvv = NUM_VVV_DEFAULT
                chooseinner = True
            elif (args.p1ry == -1) and (args.p1rx == 0) and (args.p2ry == 0):
                all_panels = P1RY_OVERSHOOT_CENTROID_LAYOUT
                num_vvv = NUM_VVV_P1RY_OVERSHOOT
            elif (args.p2ry == -1) and (args.p1rx == 0) and (args.p1ry == 0):
                all_panels = P2RY_OVERSHOOT_CENTROID_LAYOUT
                num_vvv = NUM_VVV_P2RY_OVERSHOOT
            else:
                print("invalid option for p1rx")
            if args.clustering:
                print("*** This is not implemented; don't use! ***")
                if args.pattern_center is None:
                    xc, yc = get_centroid_global(sew_out_table1)
                    args.pattern_center = [xc, yc]
                find_ring_pattern_clustering(sew_out_table1, pattern_center=args.pattern_center, radius=args.ring_rad,
                                             rad_frac=args.ring_frac, rad_tol_frac=args.ring_tol, n_rings=2,
                                             rad_inner=0.5, rad_outer=1.2, )
            else:
                if args.pattern_center is None:
                    xc, yc = get_centroid_global(sew_out_table1)
                else:
                    xc, yc = args.pattern_center[0], args.pattern_center[1]
                # print(xc, yc)
                if args.vvv_tag is not None:
                    vvv_ring_file1 = save_filename_prefix1 + "_ring_search_vvv_{}.csv".format(args.vvv_tag)
                    ring_file1 = ring_file[:-4] + "_" + args.vvv_tag + ".pdf"
                    ring_cat_file1 = ring_cat_file[:-4] + "_" + args.vvv_tag + ".txt"
                else:
                    if (args.p1ry != -1):
                        ring_file1 = ring_file[:-4] + "_P1.pdf"
                        ring_cat_file1 = ring_cat_file[:-4] + "_P1.txt"
                        vvv_ring_file1 = vvv_ring_file[:-4] + "_P1.csv"
                    else:
                        ring_file1 = ring_file[:-4] + "_P1_doubleFocusOvershoot.pdf"
                        ring_cat_file1 = ring_cat_file[:-4] + "_P1_doubleFocusOvershoot.txt"
                        vvv_ring_file1 = vvv_ring_file[:-4] + "_P1_doubleFocusOvershoot.csv"
                clast, rlast, r2std_last, sew_slice, df_slice = find_ring_pattern(sew_out_table1, all_panels=all_panels,
                                                                                  num_vvv=num_vvv,
                                                                                  chooseinner=chooseinner,
                                                                                  # pattern_center=args.pattern_center,
                                                                                  pattern_center=[xc, yc],
                                                                                  radius=args.ring_rad,
                                                                                  rad_frac=args.ring_frac, n_iter=20,
                                                                                  rad_tol_frac=args.ring_tol,
                                                                                  phase_offset_rad=args.phase_offset_rad,
                                                                                  fix_center=True, var_tol=4000, inner8=args.inner8)
                print(sew_slice)
                print(df_slice)
                if len(sew_slice) > 0:
                    plot_raw_cat(args.rawfile1, sew_slice, df=df_slice, center_pattern=clast, cropxs=cropxs,
                                 cropys=cropys, kernel_w=3, save_catlog_name=ring_cat_file1, df_LEDs=df_LEDs,
                                 save_for_vvv=vvv_ring_file1, saveplot_name=ring_file1, show=args.show,
                                 verbose=args.verbose)
                centerP1 = clast
                N_P1 = len(sew_slice)
                if os.path.exists(vvv_ring_file1) and N_P1:
                    print(
                        "Let's do a quick ring check on Panel IDs for P1S1 ring, using file {}".format(vvv_ring_file1))
                    if args.skip_p2 and args.skip_s2:
                        quick_check_raw_ring(args.rawfile1, save_for_vvv=vvv_ring_file1, labelcolor=args.labelcolor,
                                             df_LEDs=df_LEDs, saveplot_name=vvv_ring_file1[:-4] + ".png", show=True, vmax=args.ring_vmax)
                    else:
                        quick_check_raw_ring(args.rawfile1, save_for_vvv=vvv_ring_file1, labelcolor=args.labelcolor,
                                             df_LEDs=df_LEDs, saveplot_name=vvv_ring_file1[:-4] + ".png",
                                             show=False, vmax=args.ring_vmax)  # saveplot_name = vvv_ring_file[:-4] + ".png", show = args.show)
                # print("==== Center of the P1 ring is {:.2f}, {:.2f} ====".format(centerP1[0], centerP1[1]))
                # print("==== Center of the LEDs is {:.2f}, {:.2f} ====".format(center_LEDs[0], center_LEDs[1]))

                if not args.skip_p2:
                    # automatically try P2S1 ring
                    #P2S1ring_rad = 1.59 * args.ring_rad
                    P2S1ring_rad = args.P2S1ring_rad
                    if (args.p2ry != -1):
                        ring_cat_file2 = ring_cat_file[:-4] + "_P2.txt"
                        ring_file2 = ring_file[:-4] + "_P2.pdf"
                        vvv_ring_file2 = vvv_ring_file[:-4] + "_P2.csv"
                    else:
                        ring_cat_file2 = ring_cat_file[:-4] + "_P2_doubleFocusOvershoot.txt"
                        ring_file2 = ring_file[:-4] + "_P2_doubleFocusOvershoot.pdf"
                        vvv_ring_file2 = vvv_ring_file[:-4] + "_P2_doubleFocusOvershoot.csv"
                    c2, r2, r2std2, sew_slice2, df_slice2 = find_ring_pattern(sew_out_table1, all_panels=all_panels,
                                                                              num_vvv=num_vvv,
                                                                              chooseinner=False,
                                                                              # pattern_center=args.pattern_center,
                                                                              pattern_center=[xc, yc],
                                                                              radius=P2S1ring_rad,
                                                                              rad_frac=args.ring_frac, n_iter=20,
                                                                              rad_tol_frac=args.ring_tol,
                                                                              phase_offset_rad=args.phase_offset_rad,
                                                                              fix_center=True, var_tol=4000)
                    plot_raw_cat(args.rawfile1, sew_slice2, df=df_slice2, center_pattern=c2, cropxs=cropxs,
                                 cropys=cropys, kernel_w=3, save_catlog_name=ring_cat_file2, df_LEDs=df_LEDs,
                                 save_for_vvv=vvv_ring_file2, saveplot_name=ring_file2, show=args.show,
                                 verbose=args.verbose)
                    # if df_slice2 is not None:
                    N_P2 = len(sew_slice2)
                    # print(N_P2, sew_slice2)
                    # else:
                    #    N_P2 = 0
                    if args.pattern_center is None:
                        print("(diagnostic) Center of centroids weighted by flux: {:.2f} {:.2f}".format(xc, yc))
                    if os.path.exists(vvv_ring_file2):
                        print("Let's do a quick ring check on Panel IDs for P2S1 ring, using file {}".format(
                            vvv_ring_file2))
                        if args.skip_s2:
                            quick_check_raw_ring(args.rawfile1, save_for_vvv=vvv_ring_file2, labelcolor=args.labelcolor,
                                                 df_LEDs=df_LEDs, saveplot_name=vvv_ring_file2[:-4] + ".png",
                                                 show=args.show, vmax=args.ring_vmax)
                        else:
                            quick_check_raw_ring(args.rawfile1, save_for_vvv=vvv_ring_file2, labelcolor=args.labelcolor,
                                                 df_LEDs=df_LEDs, saveplot_name=vvv_ring_file2[:-4] + ".png",
                                                 show=False, vmax=args.ring_vmax)
                    print("==== Center of the LEDs is {:.2f}, {:.2f} ====".format(center_LEDs[0], center_LEDs[1]))
                if not args.skip_s2:
                    # automatically try P2S2 ring
                    #P2S2ring_rad = 2.2 * args.ring_rad
                    P2S2ring_rad = args.P2S2ring_rad
                    ring_cat_file3 = ring_cat_file[:-4] + "_S2.txt"
                    ring_file3 = ring_file[:-4] + "_S2.pdf"
                    vvv_ring_file3 = vvv_ring_file[:-4] + "_S2.csv"
                    sew_out_table3, im_med3 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                                          DETECT_MINAREA=args.DETECT_MINAREA_S2,
                                                          BACK_SIZE=args.BACK_SIZE, THRESH=args.THRESH,
                                                          DEBLEND_MINCONT=args.DEBLEND_MINCONT, sewpy_params=sew_params,
                                                          cropxs=cropxs, cropys=cropys, clean=args.clean,
                                                          savefits_name=savefits_name1, overwrite_fits=True,
                                                          saveplot_name=saveplot_name1,
                                                          savecatalog_name=savecatalog_name1, search_xs=args.search_xs,
                                                          search_ys=args.search_ys, show=False)
                    c3, r3, r2std3, sew_slice3, df_slice3 = find_ring_pattern(sew_out_table3, all_panels=all_panels,
                                                                              num_vvv=num_vvv,
                                                                              chooseinner=False,
                                                                              # pattern_center=args.pattern_center,
                                                                              pattern_center=[xc, yc],
                                                                              radius=P2S2ring_rad,
                                                                              rad_frac=args.ring_frac, n_iter=20,
                                                                              rad_tol_frac=args.ring_tol,
                                                                              phase_offset_rad=args.phase_offset_rad,
                                                                              fix_center=True, var_tol=4000)
                    plot_raw_cat(args.rawfile1, sew_slice3, df=df_slice3, center_pattern=c3, cropxs=cropxs,
                                 cropys=cropys, kernel_w=3, save_catlog_name=ring_cat_file3, df_LEDs=df_LEDs,
                                 save_for_vvv=vvv_ring_file3, saveplot_name=ring_file3, show=args.show,
                                 verbose=args.verbose)
                    # if df_slice3 is not None:
                    N_S2 = len(sew_slice3)
                    # else:
                    #    N_S2 = 0

                    # if args.pattern_center is None:
                    #    print("(diagnostic) Center of centroids weighted by flux: {:.2f} {:.2f}".format(xc, yc))
                    if os.path.exists(vvv_ring_file3):
                        print("Let's do a quick ring check on Panel IDs for P2S2 ring, using file {}".format(
                            vvv_ring_file3))
                        quick_check_raw_ring(args.rawfile1, save_for_vvv=vvv_ring_file3, labelcolor=args.labelcolor,
                                             df_LEDs=df_LEDs, saveplot_name=vvv_ring_file3[:-4] + ".png",
                                             show=args.show, vmax=args.ring_vmax)
                # print useful info at the end
                print("================")
                print("==== Center of the LEDs is {:.2f}, {:.2f} ====".format(center_LEDs[0], center_LEDs[1]))
                print("LED centroids saved in file {}".format(LED_filename))
                print("========")
                print(" Found {} P1s ".format(N_P1))
                print(" Center of the P1 ring is {:.2f}, {:.2f} ".format(centerP1[0], centerP1[1]))
                dp1x = centerP1[0] - center_LEDs[0]
                dp1y = centerP1[1] - center_LEDs[1]
                print(" Offset between center of the P1 ring and center of the LEDs is {:.2f} pix, {:.2f} pix ".format(
                    dp1x, dp1y))
                if not args.skip_p2:
                    print("========")
                    print(" Found {} P2s ".format(N_P2))
                    print(" Center of the P2 ring is {:.2f}, {:.2f} ".format(c2[0], c2[1]))
                    dp2x = c2[0] - centerP1[0]
                    dp2y = c2[1] - centerP1[1]
                    print(" Offset between center of the P2 ring and P1 ring is {:.2f} pix, {:.2f} pix".format(dp2x,
                        dp2y))
                if not args.skip_s2:
                    print("========")
                    print(" Found {} S2s ".format(N_S2))
                    print(" Center of the S2 ring is {:.2f} pix, {:.2f} pix".format(c3[0], c3[1]))
                    dp3x = c3[0] - centerP1[0]
                    dp3y = c3[1] - centerP1[1]
                    print(" Offset between center of the S2 ring and P1 ring is {:.2f} pix, {:.2f} pix".format(dp3x,
                        dp3y))

        exit(0)

    elif args.quick_ring_check is not None:
        print("doing a quick check on Panel IDs, using file {}".format(args.quick_ring_check))
        quick_check_raw_ring(args.rawfile1, save_for_vvv=args.quick_ring_check, labelcolor=args.labelcolor,
                             saveplot_name=args.quick_ring_check[:-4] + ".png", show=args.show, vmax=args.ring_vmax)

    else:
        sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w, DETECT_MINAREA=args.DETECT_MINAREA,
                                              THRESH=args.THRESH, DEBLEND_MINCONT=args.DEBLEND_MINCONT,
                                              BACK_SIZE=args.BACK_SIZE, sewpy_params=sew_params, cropxs=cropxs,
                                              cropys=cropys, clean=args.clean, savefits_name=savefits_name1,
                                              overwrite_fits=True, saveplot_name=None,
                                              savecatalog_name=savecatalog_name1, search_xs=args.search_xs,
                                              search_ys=args.search_ys)
        sew_out_table2, im_med2 = process_raw(args.rawfile2, kernel_w=args.kernel_w, DETECT_MINAREA=args.DETECT_MINAREA,
                                              THRESH=args.THRESH, DEBLEND_MINCONT=args.DEBLEND_MINCONT,
                                              BACK_SIZE=args.BACK_SIZE, sewpy_params=sew_params, cropxs=cropxs,
                                              cropys=cropys, clean=args.clean, savefits_name=savefits_name2,
                                              overwrite_fits=True, saveplot_name=None,
                                              savecatalog_name=savecatalog_name2, search_xs=args.search_xs,
                                              search_ys=args.search_ys)

        naive_comparison(sew_out_table1, sew_out_table2, im_med1, im_med2, min_dist=args.min_dist,
                         diffcat1=diffcatalog_name1, diffcat2=diffcatalog_name2, outfile1=saveplot_name1,
                         outfile2=saveplot_name2, cropxs=cropxs, cropys=cropys, verbose=args.verbose,
                         center=np.array(args.center))

        plot_diff_labelled(args.rawfile1, args.rawfile2, diffcatalog_name1, diffcatalog_name2, ind1=None, ind2=None,
                           motion_outfile_prefix=motion_outfile_prefix, outfile1=diffplot_name1,
                           outfile2=diffplot_name2, cropxs=cropxs, cropys=cropys, center=np.array(args.center))

        grid2gif(diffplot_name1, diffplot_name2, gifname)


if __name__ == '__main__':
    main()
