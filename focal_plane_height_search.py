import matplotlib.pyplot as plt
import numpy as np


P2s = [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228,
       1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328,
       1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428,
       1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128]

P1s = [1211, 1212, 1213, 1214,
       1311, 1312, 1313, 1314,
       1411, 1412, 1413, 1414,
       1111, 1112, 1113, 1114]

S1s = [2111, 2112,
       2211, 2212,
       2311, 2312,
       2411, 2412]

S2s = [2121,2122,2123,2124,
       2221,2222,2223,2224,
       2321,2322,2323,2324,
       2421,2422,2423,2424]

sectorDict = {'P1':P1s, 'P2':P2s, 'S2':P1s}

data_dir='./data/FP_search'
#Timestamp matching the number of layers added on the central module to 'move' the focal plane. Each layer is 5mm.
layers_to_times_dict={"2019_12_13_22_06_23":0,
                      "2019_12_13_22_22_26":1,
                      "2019_12_13_22_55_27":2,
                      "2019_12_13_23_14_24":3,
                      "2019_12_13_23_41_36":4}

def get_sewpy_data(csv_file):
    data=np.genfromtxt(csv_file, delimiter=',',skip_header=True)
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
        phase = 2*np.pi - ((quadrant - 1) * 0.5 * np.pi + 0.5 * np.pi * (segment - 0.5) / total_segment) - (0.5)*np.pi
    else:
        phase = ((quadrant - 1) * 0.5 * np.pi + 0.5 * np.pi * (segment - 0.5) / total_segment) #original phase form
    phase_width = 0.5 * np.pi * (1.0) / total_segment
    if is_inner_ring:
        radius = 1.5
    else:
        radius = 3.0
    if ring == "S2":
        radius *= 1.5
    return radius, phase, phase_width

def plot_param_fourier_transform(time, data_dir = data_dir):
    fig_F, ax_FT = plt.subplots(figsize=(6, 5), ncols=1)
    for ring in sectorDict.keys():
        ring_filename = "res_focal_plane_" + time + "_ring_search_vvv_" + ring + ".csv"
        ring_data = get_sewpy_data(data_dir + '/' + ring_filename)
        N = len(ring_data)
        radius = np.empty(N)
        phase = np.empty(N)
        phi = 2*np.pi/N
        for i in range(N):
            radius[i], phase[i], _ = FindPanelPosition(ring_data[i, 0], ring, False)
            if ring_data[i,0] == 1111 or ring_data[i,0] == 1121:
                index_1=i
        eccentricity = np.sqrt(1. - ((ring_data[:, 5]) / (ring_data[:, 4])) ** 2)

        a0 = np.sum(eccentricity) / N
        aN = np.empty(N)
        bN = np.empty(N)
        for i in range(N):
            aN[i] = (2 / N) * np.sum(eccentricity * np.cos((i+1) * phase))
            bN[i] = (2 / N) * np.sum(eccentricity * np.sin((i+1) * phase))

        f_transform = (a0 + aN + bN)
        c = 'k'
        if ring == 'P1':
            c = 'r'
        elif ring == 'P2':
            c = 'g'
        elif ring == 'S2':
            c = 'b'
        ax_FT.plot(np.rad2deg(phase), f_transform, 'o', label='ring {}'.format(ring), color=c)
        print("For ring {}:\n\tC_0 = {}, C_1 = {}, S_1 = {} for phase i=1 ({},{})= {}".format(ring,a0, aN[index_1], bN[index_1],ring_data[index_1,0],index_1,np.rad2deg(phase[index_1])))
    ax_FT.legend()
    plt.show()

def plot_eccentricity_on_pattern(layers_to_times_dict):
    '''
    Plots the eccentricity in color onto the mirror pattern.
    '''
    for time,layer in layers_to_times_dict.items():
        fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
        for ring in sectorDict.keys():
            ring_filename = "res_focal_plane_"+time+"_ring_search_vvv_"+ring+".csv"
            ring_data = get_sewpy_data(data_dir + '/' + ring_filename)
            N = len(ring_data)
            radius = np.empty(N)
            phase = np.empty(N)
            for i in range(N):
                radius[i], phase[i], _ = FindPanelPosition(ring_data[i,0], ring, False)
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
            bounds = np.linspace(0,1,21)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            eccentricity = np.sqrt(1. - ((ring_data[:, 5])/ (ring_data[:, 4])) ** 2)

            scatter = plt.scatter(coor_x, coor_y, c=eccentricity,
                                  cmap=cmap,norm=norm)
            panel_string = [str(int(i)) for i in ring_data[:,0]]
            for i in range(N):
                plt.annotate(panel_string[i], (coor_x[i], coor_y[i]))
        plt.title("Layer " + str(layer))
        plt.axis('off')
        plt.tight_layout()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label('Eccentricity', rotation=270)
        plt.savefig(data_dir+"/eccentricity_layer_" + str(layer)+".png")
    plt.show()

if __name__ == '__main__':
    time='2019_12_15_00_48_33'
    res_dir='./data/'
    plot_param_fourier_transform(time,data_dir=res_dir)
    plot_eccentricity_on_pattern(layers_to_times_dict)
