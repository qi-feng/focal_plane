import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P2s = [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1421, 1422, 1423,
       1424, 1425, 1426, 1427, 1428, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128]

P1s = [1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414, 1111, 1112, 1113, 1114]

S1s = [2111, 2112, 2211, 2212, 2311, 2312, 2411, 2412]

S2s = [2121, 2122, 2123, 2124, 2221, 2222, 2223, 2224, 2321, 2322, 2323, 2324, 2421, 2422, 2423, 2424]

sectorDict = {'P1': P1s, 'P2': P2s, 'S2': P1s}

# Timestamp matching the number of layers added on the central module to 'move' the focal plane. Each layer is 5mm.
layers_to_times_dict_5mm_motion = {"2019_12_13_22_06_23": 0, "2019_12_13_22_22_26": 1, "2019_12_13_22_55_27": 2,
                                   "2019_12_13_23_14_24": 3, "2019_12_13_23_41_36": 4}

layers_to_times_prefix_dict_1mm_motion = {"2019_12_16_02_34_54": {"_M2_z_0": 0},
                                          "2019_12_16_02_37_06": {"_M2_z_1": -1},
                                          "2019_12_16_02_41_37": {"_M2_z_2": -2},
                                          "2019_12_16_02_44_38": {"_M2_z_3": -3},
                                          "2019_12_16_02_49_05": {"_M2_z_4": -4},
                                          "2019_12_16_02_55_13": {"_M2_z_0": 0},
                                          "2019_12_16_03_04_24": {"_M2_z_p1": 1},
                                          "2019_12_16_03_06_46": {"_M2_z_p2": 2},
                                          "2019_12_16_03_09_05": {"_M2_z_p3": 3},
                                          "2019_12_16_03_11_44": {"_M2_z_p4": 4}, }

layers_to_times_prefix_dict_025mm_motion = {"2019_12_17_00_32_31":{"_z-1-1"   :0  },
                                            "2019_12_17_00_35_18":{"_z-1+0.25":.25  },
                                            "2019_12_17_00_37_33":{"_z-1+0.5" :.5  },
                                            "2019_12_17_00_39_57":{"_z-1+0.75" :.75 },
                                            "2019_12_17_00_42_43":{"_z-1+1.0"  :1 },
                                            "2019_12_17_00_45_16":{"_z-1+1.25" :1.25 },
                                            "2019_12_17_00_47_37":{"_z-1+1.5"  :1.5 },
                                            "2019_12_17_00_49_44":{"_z-1+1.75":1.75  },
                                            "2019_12_17_00_52_12":{"_z-1+2.0" :2.  },
                                            "2019_12_17_00_58_22":{"_z-1+2.5" :2.5  },
                                            "2019_12_17_01_16_22":{"_z-1"      :0 },
                                            "2019_12_17_01_17_29":{"_z-1-0.25":-.25  },
                                            "2019_12_17_01_19_43":{"_z-1-0.5" :-.5  },
                                            "2019_12_17_01_22_15":{"_z-1-0.75" : -0.75},
                                            "2019_12_17_01_24_26":{"_z-1-1.0" : -1. },
                                            "2019_12_17_01_26_38":{"_z-1-1.25" :-1.25 },
                                            "2019_12_17_01_28_57":{"_z-1-1.5"  :-1.5},
                                            "2019_12_17_01_31_47":{"_z-1-1.75":-1.75}}


def get_sewpy_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


def FindPanelPosition(panel, ring, focal_plane_ordering=False):
    panel_id = str(panel)
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
    if focal_plane_ordering:
        phase = 2 * np.pi - ((quadrant - 1) * 0.5 * np.pi + 0.5 * np.pi * (segment - 0.5) / total_segment) - (
            0.5) * np.pi
    else:
        phase = ((quadrant - 1) * 0.5 * np.pi + 0.5 * np.pi * (segment - 0.5) / total_segment)  # original phase form
    phase_width = 0.5 * np.pi * (1.0) / total_segment
    if is_inner_ring:
        radius = 1.5
    else:
        radius = 3.0
    if ring == "S2":
        radius *= 1.5
    return radius, phase, phase_width


def flux_area_fourier_transform(ring_data, ring):
    N = len(ring_data)
    index_1 = 0
    for i in range(N):
        ring_data.loc[i, 'radius'], ring_data.loc[i, 'phase'], ring_data.loc[i, 'phase_width'] = FindPanelPosition(
            ring_data.iloc[i]['Panel_ID_guess'], ring, False)
        if ring_data.iloc[i]['Panel_ID_guess'] == 1111 or ring_data.iloc[i]['Panel_ID_guess'] == 1121:
            index_1 = i

    c_0 = np.sum(ring_data['FLUX_AREA']) / N
    c_n = np.empty(N)
    s_n = np.empty(N)
    flux_area_ft = pd.DataFrame({'phase': ring_data.loc[:, 'phase'], "C_0": c_0, 'C_N': c_n, 'S_N': s_n})

    for i in range(N):
        c_n[i] = (2 / N) * np.sum(ring_data['FLUX_AREA'] * np.cos((i + 1) * ring_data['phase']))
        s_n[i] = (2 / N) * np.sum(ring_data['FLUX_AREA'] * np.sin((i + 1) * ring_data['phase']))

    ring_data['first_harmonic'] = c_n[index_1] * np.cos(ring_data['phase']) + s_n[index_1] * np.sin(ring_data['phase'])
    ring_data['residuals'] = ring_data['FLUX_AREA'] - c_0 - ring_data['first_harmonic']

    return ring_data, flux_area_ft, index_1


def eccentricity_fourier_transform(ring_data, ring):
    N = len(ring_data)
    index_1 = 0
    for i in range(N):
        ring_data.loc[i, 'radius'], ring_data.loc[i, 'phase'], ring_data.loc[i, 'phase_width'] = FindPanelPosition(
            ring_data.iloc[i]['Panel_ID_guess'], ring, False)
        if ring_data.iloc[i]['Panel_ID_guess'] == 1111 or ring_data.iloc[i]['Panel_ID_guess'] == 1121:
            index_1 = i
    ring_data.loc[:, 'eccentricity'] = np.sqrt(1. - ((ring_data['B_x_KR_in_pix']) / (ring_data['A_x_KR_in_pix'])) ** 2)

    c_0 = np.sum(ring_data['eccentricity']) / N
    c_n = np.empty(N)
    s_n = np.empty(N)
    eccentricity_ft = pd.DataFrame({'phase': ring_data.loc[:, 'phase'], "C_0": c_0, 'C_N': c_n, 'S_N': s_n})

    for i in range(N):
        c_n[i] = (2 / N) * np.sum(ring_data['eccentricity'] * np.cos((i + 1) * ring_data['phase']))
        s_n[i] = (2 / N) * np.sum(ring_data['eccentricity'] * np.sin((i + 1) * ring_data['phase']))

    ring_data['first_harmonic'] = c_n[index_1] * np.cos(ring_data['phase']) + s_n[index_1] * np.sin(ring_data['phase'])
    ring_data['residuals'] = ring_data['eccentricity'] - c_0 - ring_data['first_harmonic']

    return ring_data, eccentricity_ft, index_1


def plot_eccentricity_on_pattern(layers_to_times_dict, data_dir):
    '''
    Plots the eccentricity in color onto the mirror pattern.
    '''
    for time, layer in layers_to_times_dict.items():
        fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
        for ring in sectorDict.keys():
            ring_filename = "res_focal_plane_" + time + "_ring_search_vvv_" + ring + ".csv"
            try:
                ring_data = get_sewpy_data(data_dir + '/' + ring_filename)
            except:
                continue
            N = len(ring_data)
            radius = np.empty(N)
            phase = np.empty(N)
            for i in range(N):
                radius[i], phase[i], _ = FindPanelPosition(ring_data.iloc[i]['Panel_ID_guess'], ring, False)
            coor_x = radius * np.cos(phase)
            coor_y = radius * np.sin(phase)

            import matplotlib as mpl

            # cmap=plt.cm.viridis
            cmap = plt.cm.jet
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            cmaplist[0] = (.5, .5, .5, 1.0)

            # create the new map
            cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0, 1, 21)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            eccentricity = np.sqrt(1. - ((ring_data['B_x_KR_in_pix']) / (ring_data['A_x_KR_in_pix'])) ** 2)

            scatter = plt.scatter(coor_x, coor_y, c=eccentricity, cmap=cmap, norm=norm)
            panel_string = [str(int(i)) for i in ring_data['Panel_ID_guess']]
            for i in range(N):
                plt.annotate(panel_string[i], (coor_x[i], coor_y[i]))
        plt.title("Layer " + str(layer))
        plt.axis('off')
        plt.tight_layout()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label('Eccentricity', rotation=270)
        plt.savefig(data_dir + "/eccentricity_layer_" + str(layer) + ".png")
    plt.show()


def plot_fourier_decomp_by_z_layer(layer_dict, res_dir, param):
    for time, layer_prefix_dict in layer_dict.items():
        for prefix, layer in layer_prefix_dict.items():
            for ring in sectorDict.keys():
                ring_filename = "res_focal_plane_" + time + "_ring_search_vvv_" + ring + prefix + ".csv"
                ring_filepath = res_dir + '/' + ring_filename
                if os.path.exists(ring_filepath):
                    ring_data = get_sewpy_data(ring_filepath)
                    if param == "eccentricity":
                        param_name = "eccentricity"
                        ring_data, eccentricity, first_harmonic_index = eccentricity_fourier_transform(ring_data, ring)
                    elif param == "FLUX_AREA":
                        param_name = "flux_area"
                        ring_data, flux_area, first_harmonic_index = flux_area_fourier_transform(ring_data, ring)
                    c = 'k'
                    if ring == 'P1':
                        c = 'r'  # continue
                    elif ring == 'P2':
                        c = 'b'  # continue
                    elif ring == 'S2':
                        c = 'g'  # continue
                    plt.figure()
                    ax = plt.gca()
                    ring_data.plot(kind='scatter', x='phase', y='first_harmonic', color=c,
                                   label=ring + " first harmonic", ax=ax)
                    ring_data.plot(kind='scatter', x='phase', y='residuals', color=c, label=ring + " residual", ax=ax,
                                   marker='+')
                    plt.title("{} layer Z + {}".format(param_name, layer))
                    plt.xlabel("Phase (rad)")
                else:
                    continue
                plt.savefig("{}/{}_fourier_decomp_{}{}".format(res_dir, param_name, ring, prefix))
                plt.show()


def plot_avg_area_by_z_layer(layer_dict, res_dir):
    for ring in sectorDict.keys():
        if ring == "P2":
            continue
        if ring == "P1":
            flux_area_avg = np.empty([15])
            layer_value_array = np.empty([15])
        elif ring == "S2":
            flux_area_avg = np.empty([14])
            layer_value_array = np.empty([14])
        i = 0
        for time, layer_prefix_dict in layer_dict.items():
            if time == "2019_12_16_02_55_13":
                continue
            for prefix, layer in layer_prefix_dict.items():
                ring_filename = "res_focal_plane_" + time + "_ring_search_vvv_" + ring + prefix + ".csv"
                ring_filepath = res_dir + '/' + ring_filename
                if os.path.exists(ring_filepath):
                    ring_data = get_sewpy_data(ring_filepath)
                    ring_data, flux_area, first_harmonic_index = flux_area_fourier_transform(ring_data, ring)
                    flux_area_avg[i] = flux_area['C_0'][0]
                    # print(flux_area_avg[i])
                    layer_value_array[i] = layer
                    i += 1
                else:
                    continue
        plt.figure()
        plt.plot(layer_value_array, flux_area_avg, 'o', linestyle=' ', label=ring)
        print(layer_value_array)
        print(flux_area_avg)
        plt.title("Flux Area Averages")
        plt.xlabel("M2 Motion (mm)")
        plt.legend()
        plt.ylabel("Flux Area (pix^2)")
        plt.savefig("{}/{}_avg_area ".format(res_dir, ring))
        plt.show()


def main():
    res_dir = '/Users/deividribeiro/Desktop/Focal_Plane_Search_camera_Z_motion_0.25mm/'
    # plot_fourier_decomp_by_z_layer(layers_to_times_prefix_dict_1mm_motion, res_dir, "eccentricity")
    # plot_eccentricity_on_pattern(layers_to_times_dict_5mm_motion,
    #                              data_dir='./data/Focal_Plane_Search_camera_Z_motion_5mm')
    plot_avg_area_by_z_layer(layers_to_times_prefix_dict_025mm_motion, res_dir)


if __name__ == '__main__':
    main()
