import argparse
import numpy as np
import pandas as pd
import os
import yaml




# Original GR cam lens with 8mm focal length used
#pix2mm = 0.48244
#x_corners = np.array([1210,1220,1957,1945])
#y_corners = np.array([1374, 635, 648, 1385])
#center=np.array([np.mean(x_corners), np.mean(y_corners)])
# center at ~60 deg is 1583. , 1010.5


# New GR cam lens with 16mm focal length used (2x zoom in)
#pix2mm = 0.2449
pix2mm = 0.241


#these new corners mark the central module
x_corners = np.array([1762,1761,1980,1982])
y_corners = np.array([1175,954,952,1174])
center=np.array([np.mean(x_corners), np.mean(y_corners)])
center=np.array([1871.25, 1063.75])
# center at ~-5 deg is 1871.25, 1063.75

x_corners = np.array([1782,1781,2000,2002])
y_corners = np.array([1175,954,952,1174])
#center=np.array([np.mean(x_corners), np.mean(y_corners)])
center=np.array([1891.25, 1063.75])
# center at ~60 deg is 1891.25, 1063.75
# center at ~75 deg is 1896.25, 1063.75



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
respM_file = "rx_ry_matrix_lens16mm.yaml"
pattern_file = "pattern_position_lens16mm.txt"

#data_dir = './'



def find_pattern_position(panel_id,
                          center=np.array([1117.5, 1069.5]),
                          radius_mm=np.array([8, 16]),
                          pixel_scale=0.241):
    panel_id = str(panel_id)
    radius = 0.
    phase = 0.
    is_primary = False
    is_inner_ring = False
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


def find_all_pattern_positions(all_panels = np.array([1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1321, 1322, 1323,
       1324, 1325, 1326, 1327, 1328, 1421, 1422, 1423, 1424, 1425, 1426,
       1427, 1428, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1211,
       1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414,
       1111, 1112, 1113, 1114]),
                               center=np.array([1583. , 1010.5]),
                               radius_mm = np.array([20, 40]),
                               pixel_scale = 0.48244,
                               outfile="dummy_pattern_position.txt",
                               num_vvv = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  1,  2,
        3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
                              ):
    df_pattern = pd.DataFrame({'Panel':all_panels, '#': num_vvv})
    df_pattern['Xpix'] = 0
    df_pattern['Ypix'] = 0
    for i, row in df_pattern.iterrows():
        x_, y_ = find_pattern_position(row['Panel'],
                                       center=center,
                              radius_mm = radius_mm,
                              pixel_scale = pixel_scale)
        df_pattern.loc[i, 'Xpix'] = x_
        df_pattern.loc[i, 'Ypix'] = y_
    df_pattern = df_pattern[['Panel', '#', 'Xpix', 'Ypix']]
    df_pattern.to_csv(outfile, index=False, sep="\t")
    return df_pattern


def load_rx_ry_matrix(panel, respfile=respM_file):
    with open(respfile) as f:
        respM_yaml = yaml.load(f)
    if panel in respM_yaml:
        print("Loading rx ry response matrix for panel {}".format(panel))
        respM_panel = np.array(respM_yaml[panel])
        print("Matrix is {}".format(respM_panel))
        return respM_panel
    else:
        print("Response matrix for panel {} does not exist in file {}. Exiting!".format(panel, respM_file))
        exit(0)


def yes_or_no(question):
    reply = str(raw_input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter y or n")


def save_rx_ry_matrix(panel, mat, respfile=respM_file):
    with open(respfile) as f:
        respM_yaml = yaml.load(f)
    if respM_yaml is not None:
        if panel in respM_yaml:
            print("=== Matrix already exists in yaml file {} for panel {} ====".format(respfile, panel))
            print("=== If you continue we will write a duplicate entry. ===")
    sure = yes_or_no("Are you sure to save matrix \n{} \nto panel {}? ".format(mat, panel))
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


def calc_center_rx_ry(panel, x, y, center=center):
    respM = load_rx_ry_matrix(panel)
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


def calc_pattern_rx_ry(panel, x, y, pattern_file=pattern_file):
    respM = load_rx_ry_matrix(panel)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utilities to calculate rx ry resp matrix given (dx1, dy1, rx, dx2, dy2, ry), '
                                                 'where dx1 and dy1 is the motion of centroid when panel rx is introduced, '
                                                 'and dx2 and dy2 is the motion of centroid when panel ry is introduced. '
                                                 'Or calculate motion needed to go to center and pattern position for a given panel, '
                                                 'need to provide current coordinates in camera x and y. ')
    parser.add_argument('panel',type=int)
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
    parser.add_argument('--resp_file', default=respM_file,
                        help="Yaml file name with the response matrices for rx ry.")
    parser.add_argument('--center_coord', type=float, default=center)

    args = parser.parse_args()

    did_something = False

    if args.center:
        did_something = True
        if args.x * args.y == 0:
            print("Please provide current x and y (using -x and -y options).")
            exit(0)
        rx, ry = calc_center_rx_ry(args.panel, args.x, args.y, center=args.center_coord)

    if args.pattern:
        did_something = True
        if args.x * args.y == 0:
            print("Please provide current x and y (using -x and -y options).")
            exit(0)
        rx, ry = calc_pattern_rx_ry(args.panel, args.x, args.y, pattern_file= args.pattern_file)

    if args.dx1 * args.dy1 * args.dx2 * args.dy2 * args.rx * args.ry != 0:
        did_something = True
        M_RxRy_inv = calc_mat_rxy(args.dx1 , args.dy1 , args.rx ,args.dx2 , args.dy2 , args.ry)
        print("Response matrix for Rx and Ry is: \n{}".format(M_RxRy_inv))
        save_rx_ry_matrix(args.panel, M_RxRy_inv, respfile=respM_file)


    if not did_something:
        print("Seems you are using illegal combination of arguments, busted, do nothing. ")