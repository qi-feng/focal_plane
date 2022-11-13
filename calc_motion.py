import argparse
import numpy as np
import pandas as pd
import os
import yaml
import sys
import pymysql

from datetime import datetime
if not sys.version_info.major == 3:
    print("You are running python 2...")
    input = raw_input




PIX2MM = 0.482
center=np.array([1612.2804, 1024.4423])

# hard-coded matrices
# Measurement still in progress
# Original GR cam lens used
"""
matrices = {
    1424: np.array([[-0.00952418,  0.01223637],
                    [ 0.01040722,  0.00782119]]),
    1328: np.array([[ 0.00154243,  0.01588859],
       [ 0.01504861, -0.00279689]])

}
"""
# file with rx ry resp matrix
respM_file = "M1_matirx_fast.yaml"
pattern_file = "pattern_position_lens16mm.txt"

#data_dir = './'


#Inferred from Nov 09 and June 3 2022
CM_REF = np.array([[1555.3447,  967.5003],
       [1669.3301,  967.5701],
       [1555.2328, 1081.3055],
       [1669.214 , 1081.3932]], dtype='float32')

#2022-11-09 @-5 deg EL; New Alibaba 8mm lens very slightly different plate scale
LED_REF_minus5 = np.array([
    [787.7784, 1421.6349],
    [787.7224, 655.7045],
    [1230.2179, 1859.71],
    [1227.9833, 217.8342],
    [1990.3311, 1853.280],
    [1985.5582, 221.9172],
    [2422.1157, 1413.521],
    [2419.3499, 658.1667]], dtype='float32')

LED_REF = LED_REF_minus5


P2s = [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
       1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
       1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
       1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128]

P1s = [1111, 1112, 1113, 1114,
       1211, 1212, 1213, 1214,
       1311, 1312, 1313, 1314,
       1411, 1412, 1413, 1414]

S1s = [2111, 2112,
       2211, 2212,
       2311, 2312,
       2411, 2412]

S2s = [2121,2122,2123,2124,
       2221,2222,2223,2224,
       2321,2322,2323,2324,
       2421,2422,2423,2424]


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


sectorDict = {'P1':P1s, 'P2':P2s, 'S1':S1s, 'S2':S2s,
              'M1':P1s+P2s, 'M2': S1s+S2s, 'All':P1s+P2s+S1s+S2s}


string_for_deltacoords = {1111:'Panel: (11003, opc.tcp://172.17.1.110:4840, Panel_1111, 1111)',
1112:'Panel: (11017, opc.tcp://172.17.1.113:4840, Panel_1112, 1112)',
1113:'Panel: (11002, opc.tcp://172.17.1.116:4840, Panel_1113, 1113)',
1114:'Panel: (11018, opc.tcp://172.17.1.119:4840, Panel_1114, 1114)',
1121:'Panel: (12004, opc.tcp://172.17.1.111:4840, Panel_1121, 1121)',
1122:'Panel: (12011, opc.tcp://172.17.1.112:4840, Panel_1122, 1122)',
1123:'Panel: (12019, opc.tcp://172.17.1.114:4840, Panel_1123, 1123)',
1124:'Panel: (12006, opc.tcp://172.17.1.115:4840, Panel_1124, 1124)',
1125:'Panel: (12033, opc.tcp://172.17.1.117:4840, Panel_1125, 1125)',
1126:'Panel: (12022, opc.tcp://172.17.1.118:4840, Panel_1126, 1126)',
1127:'Panel: (12032, opc.tcp://172.17.1.120:4840, Panel_1127, 1127)',
1128:'Panel: (12010, opc.tcp://172.17.1.121:4840, Panel_1128, 1128)',
1211:'Panel: (11001, opc.tcp://172.17.1.122:4840, Panel_1211, 1211)',
1212:'Panel: (11006, opc.tcp://172.17.1.125:4840, Panel_1212, 1212)',
1213:'Panel: (11009, opc.tcp://172.17.1.128:4840, Panel_1213, 1213)',
1214:'Panel: (11013, opc.tcp://172.17.1.131:4840, Panel_1214, 1214)',
1221:'Panel: (12016, opc.tcp://172.17.1.123:4840, Panel_1221, 1221)',
1222:'Panel: (12034, opc.tcp://172.17.1.124:4840, Panel_1222, 1222)',
1223:'Panel: (12018, opc.tcp://172.17.1.126:4840, Panel_1223, 1223)',
1224:'Panel: (12039, opc.tcp://172.17.1.127:4840, Panel_1224, 1224)',
1225:'Panel: (12037, opc.tcp://172.17.1.129:4840, Panel_1225, 1225)',
1226:'Panel: (12035, opc.tcp://172.17.1.130:4840, Panel_1226, 1226)',
1227:'Panel: (12031, opc.tcp://172.17.1.132:4840, Panel_1227, 1227)',
1228:'Panel: (12028, opc.tcp://172.17.1.133:4840, Panel_1228, 1228)',
1311:'Panel: (11015, opc.tcp://172.17.1.134:4840, Panel_1311, 1311)',
1312:'Panel: (11005, opc.tcp://172.17.1.137:4840, Panel_1312, 1312)',
1313:'Panel: (11004, opc.tcp://172.17.1.140:4840, Panel_1313, 1313)',
1314:'Panel: (11007, opc.tcp://172.17.1.143:4840, Panel_1314, 1314)',
1321:'Panel: (12024, opc.tcp://172.17.1.135:4840, Panel_1321, 1321)',
1322:'Panel: (12017, opc.tcp://172.17.1.136:4840, Panel_1322, 1322)',
1323:'Panel: (12023, opc.tcp://172.17.1.138:4840, Panel_1323, 1323)',
1324:'Panel: (12001, opc.tcp://172.17.1.139:4840, Panel_1324, 1324)',
1325:'Panel: (12030, opc.tcp://172.17.1.141:4840, Panel_1325, 1325)',
1326:'Panel: (12041, opc.tcp://172.17.1.142:4840, Panel_1326, 1326)',
1327:'Panel: (12026, opc.tcp://172.17.1.144:4840, Panel_1327, 1327)',
1328:'Panel: (12027, opc.tcp://172.17.1.145:4840, Panel_1328, 1328)',
1411:'Panel: (11020, opc.tcp://172.17.1.146:4840, Panel_1411, 1411)',
1412:'Panel: (11011, opc.tcp://172.17.1.149:4840, Panel_1412, 1412)',
1413:'Panel: (11008, opc.tcp://172.17.1.152:4840, Panel_1413, 1413)',
1414:'Panel: (11019, opc.tcp://172.17.1.155:4840, Panel_1414, 1414)',
1421:'Panel: (12025, opc.tcp://172.17.1.147:4840, Panel_1421, 1421)',
1422:'Panel: (12014, opc.tcp://172.17.1.148:4840, Panel_1422, 1422)',
1423:'Panel: (12002, opc.tcp://172.17.1.150:4840, Panel_1423, 1423)',
1424:'Panel: (12020, opc.tcp://172.17.1.151:4840, Panel_1424, 1424)',
1425:'Panel: (12012, opc.tcp://172.17.1.153:4840, Panel_1425, 1425)',
1426:'Panel: (12015, opc.tcp://172.17.1.154:4840, Panel_1426, 1426)',
1427:'Panel: (12007, opc.tcp://172.17.1.156:4840, Panel_1427, 1427)',
1428:'Panel: (12009, opc.tcp://172.17.1.157:4840, Panel_1428, 1428)'}


def find_pattern_position(panel_id,
                          center=np.array([1612.2804, 1024.4423]),
                          radius_mm=np.array([30, 50]),
                          pixel_scale=0.482):
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

    if is_inner_ring:
        radius = radius_mm[0]
        radius_pix = radius / pixel_scale
    else:
        radius = radius_mm[1]
        radius_pix = radius / pixel_scale
    cx = center[0] - radius_pix * np.sin(phase)
    cy = center[1] + radius_pix * np.cos(phase)

    # return radius, phase
    return cx, cy


def find_all_pattern_positions(all_panels=DEFAULT_CENTROID_LAYOUT, center=np.array([1612.2804, 1024.4423]),
                               radius_mm=np.array([30, 50]), phase_offset_rad=0,
                               # clockwise is positive, about 0.2 per outer panel
                               pixel_scale=PIX2MM, outfile="dummy_pattern_position.txt", num_vvv=NUM_VVV_DEFAULT, inner8=False):
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


def load_rx_ry_matrix(panel, respfile=respM_file, verbose=False):

    with open(respfile) as f:
        respM_yaml = yaml.safe_load(f)

    #print(respM_yaml)
    #print(type(panel))

    if panel in respM_yaml:
        respM_panel = np.array(respM_yaml[panel])
        if verbose:
            print("Loading rx ry response matrix for panel {} in file {}".format(panel, respfile))
            print("Matrix is {}".format(respM_panel))
        return respM_panel
    else:
        print("Response matrix for panel {} does not exist in file {}. Exiting!".format(panel, respfile))
        exit(0)


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter y or n")


def save_rx_ry_matrix(panel, mat, respfile=respM_file, force=False):
    if os.path.exists(respfile):
        with open(respfile) as f:
            respM_yaml = yaml.safe_load(f)
    else:
        respM_yaml = None
    if respM_yaml is not None:
        if panel in respM_yaml:
            print("=== Matrix already exists in yaml file {} for panel {} ====".format(respfile, panel))
            print("=== If you continue we will write a duplicate entry. ===")
    if not force:
        sure = yes_or_no("Are you sure to save matrix \n{} \nto panel {}? ".format(mat, panel))
    else:
        sure=True
    if sure:
        with open(respfile, 'a') as f:
            yaml.dump({panel: mat.tolist()}, f)
    else:
        print("Okay, mission abort.")
    return


def load_pattern(panel, pattern_file = pattern_file):
    pattern_positions = pd.read_csv(pattern_file, sep=r"\s+")
    slice = pattern_positions[(pattern_positions.Panel == panel)]
    x = slice.Xpix.values[0]
    y = slice.Ypix.values[0]
    print("Pattern position for panel {} is \n x = {:.3f}, y={:.3f}".format(panel, x, y))
    return x, y


def calc_rx_ry(dpx, dpy, M_RxRy_inv):
    rx, ry = np.matmul(M_RxRy_inv, np.array([dpx, dpy]))
    return rx, ry


def calc_mat_rx(dpx, dpy, rx):
    M14 = dpx/rx
    M24 = dpy/rx
    M_Rx = np.array([M14, M24])
    return M_Rx


def calc_mat_rxy(dpx, dpy, rx, dpx1, dpy1, ry):
    Mx = calc_mat_rx(dpx, dpy, rx)
    My = calc_mat_rx(dpx1, dpy1, ry)
    M_RxRy = np.vstack([Mx, My])
    M_RxRy_inv = np.linalg.inv(M_RxRy)
    #return M_RxRy, M_RxRy_inv
    #print("")
    return M_RxRy_inv


def calc_center_rx_ry(panel, x, y, center=center, resp=respM_file):
    respM = load_rx_ry_matrix(panel, resp)
    dx = center[0] - x
    dy = center[1] - y
    rx, ry = calc_rx_ry(dx, dy, respM)
    print("The motion to move panel {} \nfrom x={:.3f}, y={:.3f} \nto the center {:.3f}, {:.3f} is \nrx = {:.4f}, ry = {:.4f}".format(panel,
                                                                                                        x,
                                                                                                        y,
                                                                                                        center[0],
                                                                                                        center[1],
                                                                                                        rx,
                                                                                                        ry))
    return rx, ry


def calc_pattern_rx_ry(panel, x, y, pattern_file=pattern_file, resp=respM_file):
    respM = load_rx_ry_matrix(panel, resp)
    xtarg, ytarg = load_pattern(panel, pattern_file=pattern_file)
    dx = xtarg - x
    dy = ytarg - y
    rx, ry = calc_rx_ry(dx, dy, respM)
    print("The motion to move panel {} \nfrom x={:.3f}, y={:.3f} \nto the pattern position {:.3f}, {:.3f} is \nrx = {:.4f}, ry = {:.4f}".format(panel,
                                                                                                        x,
                                                                                                        y,
                                                                                                        xtarg,
                                                                                                        ytarg,
                                                                                                        rx,
                                                                                                        ry))
    return rx, ry


def calc_center_to_pattern_rx_ry(panel, center=center, pattern_file=None, resp=respM_file,
                                 pattern_radius=0, verbose=True):
    if pattern_file is None and pattern_radius == 0:
        print("You'll need to either tell me a pattern file or the radius of the pattern to know c2p")
    respM = load_rx_ry_matrix(panel, respfile=resp)
    if pattern_radius:
        xtarg, ytarg = find_pattern_position(panel,
                              center=np.array(center),
                              radius_mm=np.array([pattern_radius, pattern_radius]),
                              pixel_scale=0.482)
    elif pattern_file is not None:
        xtarg, ytarg = load_pattern(panel, pattern_file=pattern_file)

    dx = - center[0] + xtarg
    dy = - center[1] + ytarg
    rx, ry = calc_rx_ry(dx, dy, respM)
    if verbose:
        print("The motion to move panel {} \nfrom center x={:.3f}, y={:.3f} \nto pattern {:.3f}, {:.3f} is \nrx = {:.4f}, ry = {:.4f}".format(panel,
                                                                                                        center[0],
                                                                                                        center[1],
                                                                                                        xtarg,
                                                                                                        ytarg,
                                                                                                        rx,
                                                                                                        ry))
    return rx, ry


def calc_pattern_to_center_rx_ry(panel, center=center, pattern_file=None, resp=respM_file,
                                 pattern_radius=0, verbose=True):
    if pattern_file is None and pattern_radius == 0:
        print("You'll need to either tell me a pattern file or the radius of the pattern to know p2c")
    respM = load_rx_ry_matrix(panel, respfile=resp)
    if pattern_radius:
        xtarg, ytarg = find_pattern_position(panel,
                                             center=np.array(center),
                                             radius_mm=np.array([pattern_radius, pattern_radius]),
                                             pixel_scale=0.482)
    elif pattern_file is not None:
        xtarg, ytarg = load_pattern(panel, pattern_file=pattern_file)

    dx = center[0] - xtarg
    dy = center[1] - ytarg
    rx, ry = calc_rx_ry(dx, dy, respM)
    if verbose:
        print(
        "The motion to move panel {} \nfrom center x={:.3f}, y={:.3f} \nto pattern {:.3f}, {:.3f} is \nrx = {:.4f}, ry = {:.4f}".format(
            panel,
            center[0],
            center[1],
            xtarg,
            ytarg,
            rx,
            ry))
    return rx, ry


def ring_operation_find_matrix(f1, f2, f3,
                               ry=+0.5, rx=-0.5,
                               out_matrix_file="fast_matrix.yaml",
                               force=False, center=None,
                               sanity_check=True, sanity_tol=0.2):
    print("\033[0m")
    print("\033[0;31m##############################################################")
    print("\033[0;31m==== !!! Have to be in order Ry motion image, Ref image, and Rx motion !!! ====")
    print("\033[0;31m==== !!! Make sure that only Ry motions are executed between file 1 and 2  !!! ====")
    print("\033[0;31m==== !!!            and only Rx motions are executed between file 2 and 3  !!! ====")
    print("\033[0;31m##############################################################")
    print("\033[0m")

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
    df1 = df1[['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE']]
    df2 = df2[['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE']]
    df3 = df3[['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE']]
    df3 = df3.rename(columns={"X_IMAGE": "X_IMAGE_3", "Y_IMAGE": "Y_IMAGE_3"})

    df_m = pd.merge(df1, df2, on=['Panel_ID_guess', '#'], suffixes=('_1', '_2'))
    df_m = pd.merge(df_m, df3, on=['Panel_ID_guess', '#'])

    df_m['dX_pix_motion1'] = df_m['X_IMAGE_2'] - df_m['X_IMAGE_1']
    df_m['dY_pix_motion1'] = df_m['Y_IMAGE_2'] - df_m['Y_IMAGE_1']

    df_m['dX_pix_motion2'] = df_m['X_IMAGE_3'] - df_m['X_IMAGE_2']
    df_m['dY_pix_motion2'] = df_m['Y_IMAGE_3'] - df_m['Y_IMAGE_2']

    # df_m['Matrix']=0

    for i, row in df_m.iterrows():
        print("================")
        M_RxRy_inv = calc_mat_rxy(row['dX_pix_motion2'], row['dY_pix_motion2'], rx,
                                  row['dX_pix_motion1'], row['dY_pix_motion1'], ry)
        # df_m['Matrix'] = M_RxRy_inv
        print("Panel {} Response matrix for Rx and Ry is: \n{}".format(row['Panel_ID_guess'], M_RxRy_inv))
        if out_matrix_file is not None:
            save_rx_ry_matrix(int(row['Panel_ID_guess']), M_RxRy_inv, respfile=out_matrix_file, force=force)
        # sanity check
        if sanity_check:
            rx_sanity, ry0 = calc_rx_ry(row['dX_pix_motion2'], row['dY_pix_motion2'], M_RxRy_inv)
            rx0, ry_sanity = calc_rx_ry(row['dX_pix_motion1'], row['dY_pix_motion1'], M_RxRy_inv)
            print("Recovered motions are: ry = {}, rx = {}".format(ry_sanity, rx_sanity))
            print("Input motions are: ry = {}, rx = {}".format(ry, rx))
            if abs((rx_sanity-rx)/rx) <= sanity_tol and abs((ry_sanity-ry)/ry) <= sanity_tol:
                print("Sanity check passed! Looking good. ")
            else:
                print("\033[0m")
                print("\033[0;33m##############################################################")
                print("\033[0;33m==== !!!Sanity check failed! Not looking good.  !!! ====")
                print("\033[0;33m==== !!!Your tolerence for insanity is {}%  !!! ====".format(sanity_tol*100))
                print("\033[0;33m##############################################################")
                print("\033[0m")

        if center is not None:
            dx = row['X_IMAGE_1'] - center[0]
            dy = row['Y_IMAGE_1'] - center[1]
            rxc, ryc = calc_rx_ry(dx, dy, M_RxRy_inv)
            print(row['X_IMAGE_1'], row['Y_IMAGE_1'])
            print(dx, dy)
            print("Motion from center {},{} to position in file 1 is rx = {}, ry = {}".format(center[0], center[1], rxc, ryc))

    return df_m


def calc_r2c_from_file(f,resp_file="fast_matrix.yaml",center=None,outfile=None):
    df1 = pd.read_csv(f)
    df1 = df1[['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE']]
    if center is None:
        center = [np.mean(df1['X_IMAGE']), np.mean(df1['Y_IMAGE'])]

    now_ = datetime.now()
    if outfile is not None:
        fout = open(outfile, "w")
        fout.write("Mirror: (1, 1, PrimaryMirror, 1)\n")
        fout.write("Timestamp: {}\n".format(now_.ctime()))
        fout.write("Global coordinates:\n")
        for i in range(6):
            fout.write("0.00E+00\n")

    move_dict = {}
    for i, row in df1.iterrows():
        panel = row['Panel_ID_guess']
        #print("================")

        dx = - row['X_IMAGE'] + center[0]
        dy = - row['Y_IMAGE'] + center[1]
        print("===={}====".format(panel))
        M_RxRy_inv = load_rx_ry_matrix(panel, respfile=resp_file)
        rxc, ryc = calc_rx_ry(dx, dy, M_RxRy_inv)
        #print(row['X_IMAGE'], row['Y_IMAGE'])
        #print(dx, dy)
        print("Motion from X={} Y={} to center {},{} is rx = {}, ry = {}".format(row['X_IMAGE'], row['Y_IMAGE'], center[0], center[1], rxc, ryc))
        move_dict[panel] = [rxc, ryc]

    if outfile is not None:
        for k_ in string_for_deltacoords.keys():
            fout.write("****************************************\n")
            fout.write(string_for_deltacoords[k_])
            fout.write("\n")
            fout.write("0\n")
            fout.write("0\n")
            if k_ in move_dict.keys():
                fout.write("z need to implement\n")
                fout.write("{}\n".format(move_dict[k_][0]))
                fout.write("{}\n".format(move_dict[k_][1]))
            else:
                fout.write("0\n")
                fout.write("0\n")
                fout.write("0\n")
            fout.write("0\n")
    if outfile is not None:
        fout.close()


# just for mirrordeltacoords
class pSCTDB_readonly:
    def __init__(self, host='romulus.ucsc.edu'):
        self.DB_HOST = host
        self.DB_USER = os.getenv('CTADBUSERREADONLY')
        self.DB_PASSWD = os.getenv('CTADBPASSREADONLY')
        self.DB_ONLINE = os.getenv('CTAONLINEDB')
        self.DB_OFFLINE = 'CTAoffline'
        self.DB_PORT = int(os.getenv('CTADBPORT'))

    def connect(self):
        try:
            self.conn = pymysql.connect(host=self.DB_HOST,
                                        user=self.DB_USER,
                                        passwd=self.DB_PASSWD,
                                        db=self.DB_ONLINE,
                                        port=self.DB_PORT)
            self.cur = self.conn.cursor()
        except:
            print("Cannot connect to {}, consider changing this".format(self.DB_HOST))
            exit()

    def close_connect(self):
        self.cur.close()
        self.conn.close()

    def do_query(self, query):
        self.connect()
        nentries = self.cur.execute(query)
        # self.close_connect()
        return self.cur

    def get_panel_info(self, panels):
        if not isinstance(panels, list):
            panels = [str(panels)]
        else:
            panels = [str(i) for i in panels]
        sers = []
        ips = []
        self.connect()
        for p_ in panels:
            #print("====")
            query = "select serial_number, mpcb_ip_address from Opt_MPMMapping where position = {} and end_date is NULL".format(
                p_)
            nentries = self.cur.execute(query)
            for i, row in enumerate(self.cur):
                # ips.append(row[0])
                #print("Panel {} serial {}".format(p_, row[0]))
                #print("Panel {} IP {}".format(p_, row[1]))
                sers.append(row[0])
                ips.append(row[1])
            #print("====")
        self.close_connect()
        return sers, ips


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utilities to calculate rx ry resp matrix given (dx1, dy1, rx, dx2, dy2, ry), '
                                                 'where dx1 and dy1 is the motion of centroid when panel rx is introduced, '
                                                 'and dx2 and dy2 is the motion of centroid when panel ry is introduced. '
                                                 'Or calculate motion needed to go to center and pattern position for a given panel, '
                                                 'need to provide current coordinates in camera x and y. ')
    parser.add_argument('--panel',type=int)
    parser.add_argument('-f', '--files', nargs = 3, default=[None, None, None],
                        help="Csv file names, it has to b 3 files, with the pattern positions."
                             "The sequence has to be: file1 -> Ry motion -> file2 -> Rx motion -> file3.")

    parser.add_argument('--dx1',type=float, default=0)
    parser.add_argument('--dy1',type=float, default=0)
    parser.add_argument('--rx', type=float, default=0)
    parser.add_argument('--dx2', type=float, default=0)
    parser.add_argument('--dy2', type=float, default=0)
    parser.add_argument('--ry', type=float, default=0)
    parser.add_argument('-c','--center', dest="center", action='store_true')
    parser.add_argument('-p', '--pattern', dest="pattern", action='store_true')
    parser.add_argument('-x',type=float, default = 0, help="current x if calculating motion to achieve center or pattern")
    parser.add_argument('-y',type=float, default = 0, help="current x if calculating motion to achieve center or pattern")
    parser.add_argument('--pattern_file', default=pattern_file,
                        help="Text file name with the pattern positions.")
    parser.add_argument('--resp_file', default=None,
                        help="Yaml file name with the response matrices for rx ry.")
    parser.add_argument('--force', action='store_true', help="This will force write to resp_file without asking.")
    parser.add_argument('--center_coord', nargs = 2, type = float, default=list(center))
    parser.add_argument('--dry_run', action='store_true', help="This will not attempt to write resp mat to file. ")
    parser.add_argument('--c2p', action='store_true', help="Center to pattern")
    parser.add_argument('--p2c', action='store_true', help="Pattern to center (aka ring to focus)")
    parser.add_argument('--sanity_tol',type=float, default = 0.2, help="At what fraction do you become insane?")
    parser.add_argument('--sector', default=None, help="Can choose a sector, only P1, P2 implemented for now, for c2p or p2c.")
    parser.add_argument('--z_factor', default=2, help="Factor for delta_z motion (delta_z/delta_ry). Default is 2, i.e. if delta_ry is 0.1, delta_z is 0.2")
    parser.add_argument('--pattern_radius',type=float, default = 0, help="For c2p or p2c")
    parser.add_argument('--outfile', default=None,
                        help="Text file name to write mirrordeltacoords.")
    parser.add_argument('-H', '--host', default='172.17.10.10', help="Host for DB", type=str)
    parser.add_argument('--r2cfile', default=None)
    parser.add_argument('--r2c_outfile', default=None)


    args = parser.parse_args()

    did_something = False

    if args.r2cfile is not None:
        if args.resp_file is None:
            print("Need matrix file")
            exit(0)
        calc_r2c_from_file(args.r2cfile, resp_file=args.resp_file, center=None, outfile=args.r2c_outfile)
        did_something = True

    if args.files[0] is not None:
        if len(args.files)!=3:
            print("Have to provide 3 files.")
            exit(0)
        if args.rx == 0 or args.ry==0:
            print("Ry and Rx cannot be 0.")
            exit(0)
        if args.dry_run:
            resp_file = None
        else:
            resp_file = args.resp_file
        df_mfast = ring_operation_find_matrix(
            args.files[0], args.files[1], args.files[2],
            ry=args.ry, rx=args.rx, center=args.center_coord,
            out_matrix_file=resp_file, sanity_tol=args.sanity_tol,
            force=args.force)
        print("Done")
        exit(0)


    if args.center:
        did_something = True
        print("Using resp file {}".format(args.resp_file))
        if args.x * args.y == 0:
            print("Please provide current x and y (using -x and -y options).")
            exit(0)
        rx, ry = calc_center_rx_ry(args.panel, args.x, args.y, center=args.center_coord, resp=args.resp_file)

    if args.pattern:
        did_something = True
        print("Using resp file {}".format(args.resp_file))
        if args.x * args.y == 0:
            print("Please provide current x and y (using -x and -y options).")
            exit(0)
        rx, ry = calc_pattern_rx_ry(args.panel, args.x, args.y, pattern_file= args.pattern_file, resp=args.resp_file)

    if args.c2p:
        did_something = True
        print("Using resp file {}".format(args.resp_file))
        print("calculating focus to ring")
        if args.sector is not None:
            if args.sector not in sectorDict:
                print("Illegal sector {}".format(args.sector))
            else:
                print("Sector {} selected, found panels".format(args.sector))
                if args.outfile is not None:
                    if os.path.exists(args.outfile):
                        sure = yes_or_no("Are you sure to overwrite mirrordeltacoords file \n{} ? ".format(args.outfile))
                        if not sure:
                            print("Okay, provide a different name then. Done. ")
                            exit(0)

                    with open(args.outfile, 'w') as outf:
                        if args.sector == 'P1' or args.sector == 'P2':
                            outf.write("Mirror: (1, 1, PrimaryMirror, 1)\n")
                        elif args.sector == 'S1' or args.sector == 'S2':
                            outf.write("Mirror: (2, 2, SecondaryMirror, 2)\n")
                        now = datetime.now()
                        #outf.write("Timestamp: Wed Nov 13 11:40:12 2019")
                        outf.write("Timestamp: {}\n".format(now.strftime("%Y-%m-%d %A %H:%M:%S")))
                        outf.write("Global coordinates:\n")
                        outf.write("0.00E+00\n0.00E+00\n0.00E+00\n0.00E+00\n0.00E+00\n0.00E+00\n")

                ps = sectorDict[args.sector]
                print(ps)
                testDB = pSCTDB_readonly(args.host)
                sers, ips = testDB.get_panel_info(ps)

            for i, panel in enumerate(ps):
                if args.pattern_radius > 0:
                    rx, ry = calc_center_to_pattern_rx_ry(panel, center=args.center_coord,
                                                          pattern_radius=args.pattern_radius,
                                                          resp=args.resp_file, verbose=False)
                else:
                    rx, ry = calc_center_to_pattern_rx_ry(panel, center=args.center_coord,
                                                          pattern_file=args.pattern_file,
                                                          resp=args.resp_file, verbose=False)
                print("Panel {}, rx={}, ry={}".format(panel, rx, ry))
                if args.outfile is not None:
                    with open(args.outfile, 'a') as outf:
                        outf.write("****************************************\n")
                        outf.write("Panel: ({}, opc.tcp://{}:4840, Panel_{}, {})\n".format(sers[i], ips[i], panel, panel))
                        outf.write("0\n0\n{}\n{}\n{}\n0\n".format(args.z_factor*ry, rx, ry))
                print("****************************************\n")
                print("Panel: ({}, opc.tcp://{}:4840, Panel_{}, {})".format(sers[i], ips[i], panel, panel))
                print("0\n0\n{}\n{}\n{}\n0\n".format(args.z_factor * ry, rx, ry))
        else:
            if args.pattern_radius > 0:
                calc_center_to_pattern_rx_ry(args.panel, center=args.center_coord, pattern_radius=args.pattern_radius,
                                             resp=args.resp_file, verbose=True)
            else:
                calc_center_to_pattern_rx_ry(args.panel, center=args.center_coord, pattern_file=args.pattern_file,
                                             resp=args.resp_file, verbose=True)

    if args.p2c:
        did_something = True
        print("Using resp file {}".format(args.resp_file))
        print("calculating ring to focus")
        if args.sector is not None:
            if args.sector not in sectorDict:
                print("Illegal sector {}".format(args.sector))
            else:
                print("Sector {} selected, found panels".format(args.sector))
                ps = sectorDict[args.sector]
                print(ps)
            for panel in ps:
                if args.pattern_radius > 0:
                    rx, ry = calc_pattern_to_center_rx_ry(panel, center=args.center_coord,
                                                          pattern_radius=args.pattern_radius,
                                                          resp=args.resp_file, verbose=False)
                else:
                    rx, ry = calc_pattern_to_center_rx_ry(panel, center=args.center_coord,
                                                          pattern_file=args.pattern_file,
                                                          resp=args.resp_file, verbose=False)
                print("Panel {}, rx={}, ry={}".format(panel, rx, ry))
        else:
            if args.pattern_radius > 0:
                calc_pattern_to_center_rx_ry(args.panel, center=args.center_coord, pattern_radius=args.pattern_radius,
                                             resp=args.resp_file, verbose=True)
            else:
                calc_pattern_to_center_rx_ry(args.panel, center=args.center_coord, pattern_file=args.pattern_file,
                                             resp=args.resp_file, verbose=True)


    if args.dx1 * args.dy1 * args.dx2 * args.dy2 * args.rx * args.ry != 0:
        did_something = True
        M_RxRy_inv = calc_mat_rxy(args.dx1 , args.dy1 , args.rx ,args.dx2 , args.dy2 , args.ry)
        print("Response matrix for Rx and Ry is: \n{}".format(M_RxRy_inv))
        save_rx_ry_matrix(args.panel, M_RxRy_inv, respfile=respM_file)


    if not did_something:
        print("Seems you are using illegal combination of arguments, busted, do nothing. ")