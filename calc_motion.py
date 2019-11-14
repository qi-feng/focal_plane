import argparse
import numpy as np
import pandas as pd
import os
import yaml
import sys
if not sys.version_info.major == 3:
    print("You are running python 2...")
    input = raw_input



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
#x_corners = np.array([1762,1761,1980,1982])
#y_corners = np.array([1175,954,952,1174])
#center=np.array([np.mean(x_corners), np.mean(y_corners)])
#center=np.array([1871.25, 1063.75])
# center at ~-5 deg is 1871.25, 1063.75

#x_corners = np.array([1782,1781,2000,2002])
#y_corners = np.array([1175,954,952,1174])
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


def get_central_mod_corners(center=np.array([1891.25, 1063.75]),
                            cmod_xoffset = np.array([-109.25, -110.25, 108.75, 110.75]),
                            cmod_yoffset = np.array([ 111.25, -109.75, -111.75,  110.25])):
    x_corners = cmod_xoffset + center[0]
    y_corners = cmod_yoffset + center[1]
    return x_corners, y_corners



def find_pattern_position(panel_id,
                          center=np.array([1891.25, 1063.75]),
                          radius_mm=np.array([20, 40]),
                          pixel_scale=0.241):
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


def find_all_pattern_positions(all_panels = np.array([1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1321, 1322, 1323,
       1324, 1325, 1326, 1327, 1328, 1421, 1422, 1423, 1424, 1425, 1426,
       1427, 1428, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1211,
       1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414,
       1111, 1112, 1113, 1114]),
                               center=np.array([1891.25, 1063.75]),
                               radius_mm = np.array([20, 40]),
                               pixel_scale = 0.241,
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
    print(respM_yaml)
    print(type(panel))
    print(respM_yaml[1211])

    if panel in respM_yaml:
        print("Loading rx ry response matrix for panel {} in file {}".format(panel, respfile))
        respM_panel = np.array(respM_yaml[panel])
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
            respM_yaml = yaml.load(f)
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


def calc_center_to_pattern_rx_ry(panel, center=center, pattern_file=pattern_file, resp=respM_file):
    respM = load_rx_ry_matrix(panel, respfile=resp)
    xtarg, ytarg = load_pattern(panel, pattern_file=pattern_file)
    dx = - center[0] + xtarg
    dy = - center[1] + ytarg
    rx, ry = calc_rx_ry(dx, dy, respM)
    print("The motion to move panel {} \nfrom center x={:.3f}, y={:.3f} \nto pattern {:.3f}, {:.3f} is \nrx = {:.4f}, ry = {:.4f}".format(panel,
                                                                                                        center[0],
                                                                                                        center[1],
                                                                                                        xtarg,
                                                                                                        ytarg,
                                                                                                        rx,
                                                                                                        ry))
    return rx, ry


def ring_operation_find_matrix(f1, f2, f3,
                               ry=-0.25, rx=-0.5,
                               out_matrix_file="fast_matrix.yaml",
                               force=False, center=None,
                               sanity_check=True):
    print("\033[0m")
    print("\033[0;31m##############################################################")
    print("\033[0;31m==== !!! Have to be a Ry motion first and a Rx motion after !!! ====")
    print("\033[0;31m==== !!! Make sure that only Ry motions are executed between file 1 and 2  !!! ====")
    print("\033[0;31m==== !!!            and only Rx motions are executed between file 2 and 3  !!! ====")
    print("\033[0;31m##############################################################")
    print("\033[0m")

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
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
            if abs(rx_sanity-rx)/rx <= 0.1 and abs(ry_sanity-ry)/ry <= 0.1:
                print("Sanity check passed! Looking good. ")
            else:
                print("Sanity check failed! Not looking good. ")

        if center is not None:
            dx = row['X_IMAGE_1'] - center[0]
            dy = row['Y_IMAGE_1'] - center[1]
            rxc, ryc = calc_rx_ry(dx, dy, M_RxRy_inv)
            print("Motion from center {},{} to position in file 1 is rx = {}, ry = {}".format(center[0], center[1], rxc, ryc))

    return df_m




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
    parser.add_argument('--resp_file', default=respM_file,
                        help="Yaml file name with the response matrices for rx ry.")
    parser.add_argument('--force', action='store_true', help="This will force write to resp_file without asking.")
    parser.add_argument('--center_coord', nargs = 2, type = float, default=list(center))
    parser.add_argument('--dry_run', action='store_true', help="This will not attempt to write resp mat to file. ")
    parser.add_argument('--c2p', action='store_true')

    args = parser.parse_args()

    did_something = False

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
            out_matrix_file=resp_file,
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
        calc_center_to_pattern_rx_ry(args.panel, center=args.center_coord, pattern_file=args.pattern_file, resp=args.resp_file)


    if args.dx1 * args.dy1 * args.dx2 * args.dy2 * args.rx * args.ry != 0:
        did_something = True
        M_RxRy_inv = calc_mat_rxy(args.dx1 , args.dy1 , args.rx ,args.dx2 , args.dy2 , args.ry)
        print("Response matrix for Rx and Ry is: \n{}".format(M_RxRy_inv))
        save_rx_ry_matrix(args.panel, M_RxRy_inv, respfile=respM_file)


    if not did_something:
        print("Seems you are using illegal combination of arguments, busted, do nothing. ")