import argparse

from matplotlib.patches import Ellipse
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {'size': 14}
import matplotlib

matplotlib.rc('font', **font)

try:
    import cv2
    has_cv2 = True
except:
    print("Can't import cv2!!")
    has_cv2 = False


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




def read_raw(f='./GAS_image.raw', cols=2592, rows=1944, outfile=None, show=False):
    fd = open(f, 'rb')
    f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
    im = f.reshape((rows, cols)) #notice row, column format
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


def plot_diff_labelled(rawf1, rawf2, cat1, cat2,
                       ind1=None, ind2=None,
                       motion_outfile_prefix="motion_output",
                       outfile1="new_label1.pdf", outfile2="new_label2.pdf",
                       cropxs=(1350, 1800), cropys=(1250, 800),):
    #cat1 and cat2 are the * diff * cats
    im1 = read_raw(rawf1)
    im2 = read_raw(rawf2)

    kernel_w = 3
    im1 = cv2.medianBlur(im1, kernel_w)
    im2 = cv2.medianBlur(im2, kernel_w)

    cat1 = pd.read_csv(cat1, sep=r"\s+")
    cat2 = pd.read_csv(cat2, sep=r"\s+")

    #cat1_screen = cat1[(cat1.X_IMAGE >= np.min(x_corners)) & (cat1.X_IMAGE <= np.max(x_corners))]
    #cat2_screen = cat1[(cat2.Y_IMAGE >= np.min(y_corners)) & (cat2.Y_IMAGE <= np.max(y_corners))]

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    max_pixel_crop1 = np.max(im1[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area in image 1 reaches {}".format(max_pixel_crop1))
    if max_pixel_crop1 == 255:
        #print("Image 1 is saturated. ")
        sat_pix = np.sum(im1 == 255)
        percent_satur = 100.0 * sat_pix / (cropys[0] - cropys[1]) / (cropxs[1] - cropxs[0])
        print(
            "{:.3f}% of the pixels (= {} pixels) in zoomed area of image 1 is saturated. ".format(percent_satur, sat_pix))

    ax_img = ax.imshow(im1, cmap='gray', vmax=max_pixel_crop1)
    #ax_img = ax.imshow(im1, cmap='gray')
    fig.colorbar(ax_img)

    if ind1 is not None:
        motion_outfile = motion_outfile_prefix+"ind"+str(ind1)+"_ind"+str(ind2)+".txt"

        #row1 = cat1.loc[ind1]
        row1 = cat1.loc[cat1["ID"] == ind1]
        #print(row1)
        #print(row1['X_IMAGE'].values[0])

        with open(motion_outfile, 'w') as io_:
            io_.write("# center (hard coded): ({}, {})\n".format(center[0], center[1]))
            io_.write(" ".join(cat1.columns))
            io_.write("\n")
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")
        kr = row1['KRON_RADIUS'].values[0]
        e = Ellipse(xy=np.array([row1['X_IMAGE'].values[0], row1['Y_IMAGE'].values[0]]),
                    width=row1['A_IMAGE'].values[0]*kr,
                    height=row1['B_IMAGE'].values[0]*kr,
                    angle=row1['THETA_IMAGE'].values[0],
                    linewidth=2, fill=False, )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('g')
        ax.annotate(int(row1['ID'].values[0]), xy=np.array([row1['X_IMAGE'].values[0], row1['Y_IMAGE'].values[0]]),
                    #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                    xytext=(np.array([row1['X_IMAGE'].values[0] - 80, row1['Y_IMAGE'].values[0] - 80])), # for new lens
                    color='g', alpha=0.8,
                    arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=2, headlength=4, width=1, alpha=0.7),
                    )
    else:
        for i, row1 in cat1.iterrows():
            kr = row1['KRON_RADIUS']
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE']*kr,
                        height=row1['B_IMAGE']*kr,
                        angle=row1['THETA_IMAGE'],
                        linewidth=2, fill=False, )
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])), # for new lens
                        color='g', alpha=0.8,
                        arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=2, headlength=4, width=1, alpha=0.7),
                        )

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners,
                y_corners, s=20, facecolors='none', edgecolors='m')

    plt.tight_layout()
    #plt.xlim(1170, 1970)
    #plt.ylim(1410, 610)

    plt.savefig(outfile1)

    #plt.show()

    max_pixel_crop2 = np.max(im2[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area in image 2 reaches {}".format(max_pixel_crop2))
    if max_pixel_crop2 == 255:
        sat_pix = np.sum(im2 == 255)
        percent_satur = 100.0 * sat_pix / (cropys[0] - cropys[1]) / (cropxs[1] - cropxs[0])
        print(
            "{:.3f}% of the pixels (= {} pixels) in zoomed area of image 2 is saturated. ".format(percent_satur,
                                                                                                  sat_pix))

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax_img = ax.imshow(im2, cmap='gray', vmax=max_pixel_crop2)

    fig.colorbar(ax_img)

    if ind2 is not None:
        motion_outfile = motion_outfile_prefix + "ind" + str(ind1) + "_ind" + str(ind2) + ".txt"
        #row1 = cat2.loc[ind2]
        row1 = cat2.loc[cat2['ID'] == ind2]
        with open(motion_outfile, 'a') as io_:
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")
        kr = row1['KRON_RADIUS'].values[0]
        e = Ellipse(xy=np.array([row1['X_IMAGE'].values[0], row1['Y_IMAGE'].values[0]]),
                    width=row1['A_IMAGE'].values[0]*kr,
                    height=row1['B_IMAGE'].values[0]*kr,
                    angle=row1['THETA_IMAGE'].values[0],
                    linewidth=2, fill=False, )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('y')
        ax.annotate(int(row1['ID'].values[0]), xy=np.array([row1['X_IMAGE'].values[0], row1['Y_IMAGE'].values[0]]),
                    #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                    xytext=(np.array([row1['X_IMAGE'].values[0] - 80, row1['Y_IMAGE'].values[0] - 80])), # for new lens
                    color='y',alpha=0.8,
                    arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=2, headlength=4, width=1, alpha=0.7),
                    )
    else:
        for i, row1 in cat2.iterrows():
            kr = row1['KRON_RADIUS']
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE']*kr,
                        height=row1['B_IMAGE']*kr,
                        angle=row1['THETA_IMAGE'],
                        linewidth=2, fill=False, )
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        # xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                        color='y', alpha=0.8,
                        arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=2, headlength=4, width=1,
                                        alpha=0.7),
                        )

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners,
                y_corners, s=20, facecolors='none', edgecolors='m')

    plt.tight_layout()
    #plt.xlim(1170, 1970)
    #plt.ylim(1410, 610)

    plt.savefig(outfile2)
    #plt.show()

    if ind1 is not None and ind2 is not None:
        dx, dy, dist_pix = calc_dist(cat1, cat2, ind1, ind2)
        with open(motion_outfile, 'a') as io_:
            io_.write("# dx = {:.4f} pix, dy = {:.4f} pix, distance = {:.4f} pix \n".format(dx, dy, dist_pix))


def calc_dist(cat1, cat2, n1, n2, pix2mm = 0.48244):
    dx = (cat2.loc[cat2["ID"] == n2]['X_IMAGE'].values[0] - cat1.loc[cat1["ID"] == n1]['X_IMAGE'].values[0])
    dy = (cat2.loc[cat2["ID"] == n2]['Y_IMAGE'].values[0] - cat1.loc[cat1["ID"] == n1]['Y_IMAGE'].values[0])
    dist_pix = np.sqrt(dx**2 + dy**2)
    dist_mm = dist_pix * pix2mm
    print("centroid coordinate before motion: x = {:.4f} pix y = {:.4f} pix".format(cat1.loc[cat1["ID"] == n1]['X_IMAGE'].values[0], cat1.loc[cat1["ID"] == n1]['Y_IMAGE'].values[0]))
    print("centroid coordinate after motion: x = {:.4f} pix y = {:.4f} pix".format(cat2.loc[cat2["ID"] == n2]['X_IMAGE'].values[0], cat2.loc[cat2["ID"] == n2]['Y_IMAGE'].values[0]))
    print("centroid moved in x {:.4f} pix and in y {:.4f} pix".format(dx, dy))
    print("centroid moved by distance {:.4f} pix = {:.4f} mm".format(dist_pix, dist_mm))

    return dx, dy, dist_pix


def get_datetime_rawname(raw_name):
    import re
    pattern = r'\b\w{1,4}-\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2}'
    match = re.search(pattern, raw_name)
    dt_match = match.group()  # raw_name[:match.start()]
    dt_match = "_".join(dt_match.split('-'))
    dt_match = "_".join(dt_match.split(':'))
    return dt_match


def grid2gif(im1, im2, output_gif):
    str1 = 'convert -delay 50 -loop 0 -quality 100 -density 144 ' + im1 +' ' + im2  + ' ' + output_gif
    subprocess.call(str1, shell=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare two raw images of stars at the focal plane')

    parser.add_argument('rawfile1',type=str)
    parser.add_argument('rawfile2',type=str)
    parser.add_argument('--diffcatalog_name1', default=None,
                        help="File name of the ascii catalog for sources only in the first image. "
                             "If not provided, will use default that focal_plane.py uses"
                             "i.e., res_focal_plane_YYYY_MM_DD_HH_MM_SS_diff_cat1.txt")
    parser.add_argument('--diffcatalog_name2', default=None,
                        help="File name of the ascii catalog for sources only in the second image. "
                             "If not provided, will use default that focal_plane.py uses"
                             "i.e., res_focal_plane_YYYY_MM_DD_HH_MM_SS_diff_cat2.txt")

    parser.add_argument('-1', '--ind1', dest="ind1", default=None, type=int,
                        help="Index of the centroid in the first catalog")
    parser.add_argument('-2', '--ind2', dest="ind2", default=None, type=int,
                        help="Index of the centroid in the second catalog")
    parser.add_argument('-o', '--motion_outfile_prefix', dest="motion_outfile_prefix", default=None,
                        help="File name prefix of the output catalog to calculate motion.")
    parser.add_argument('--gifname', default=None,
                        help="File name to save gif animation. ")

    parser.add_argument('--saveplot_name1',default=None,
                        help="File name of the image (jpeg or pdf etc) for the first image.")
    parser.add_argument('--saveplot_name2', default=None,
                        help="File name of the image (jpeg or pdf etc) for the second image if you want to choose.")

    parser.add_argument('--datadir', default="data",
                        help="Folder to save all output files. Default is ./data (ignored by git)")

    #plt.xlim(1050, 2592)
    #plt.ylim(1850,250)



    parser.add_argument('--cropx1',
                        default=1050,
                        #default=1170,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")
    parser.add_argument('--cropx2',
                        default=2592,
                        #default=1970,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")

    parser.add_argument('--cropy1',
                        #default=1410,
                        default=1850,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")
    parser.add_argument('--cropy2',
                        #default=610,
                        default=250,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")

    args = parser.parse_args()

    cropxs = (args.cropx1, args.cropx2)
    cropys = (args.cropy1, args.cropy2)

    dt_match1 = get_datetime_rawname(args.rawfile1)
    save_filename_prefix1 = os.path.join(args.datadir,"res_focal_plane_" + dt_match1)
    dt_match2 = get_datetime_rawname(args.rawfile2)
    save_filename_prefix2 = os.path.join(args.datadir,"res_focal_plane_" + dt_match2)

    if args.diffcatalog_name1 is None:
        diffcatalog_name1 = save_filename_prefix1 + "_diff_cat1.txt"
        print("Using default catalog 1 file name {}! Make sure this is what you want".format(diffcatalog_name1))
    else:
        diffcatalog_name1 = os.path.join(args.datadir,args.diffcatalog_name1)
    if args.saveplot_name1 is None:
        diffplot_name1 = save_filename_prefix1 + "_find_diff_cat1.pdf"
        print("Using default file name {} to save diff cat1 plot! Make sure this is what you want".format(diffplot_name1))
    else:
        diffplot_name1 = os.path.join(args.datadir,args.diffplot_name1)
    if args.diffcatalog_name2 is None:
        diffcatalog_name2 = save_filename_prefix2 + "_diff_cat2.txt"
        print("Using default catalog 2 file name {}! Make sure this is what you want".format(diffcatalog_name2))
    else:
        diffcatalog_name2 = os.path.join(args.datadir,args.diffcatalog_name2)
    if args.saveplot_name1 is None:
        diffplot_name2 = save_filename_prefix2 + "_find_diff_cat2.pdf"
        print("Using default file name {} to save diff cat2 plot! Make sure this is what you want".format(diffplot_name2))
    else:
        diffplot_name2 = os.path.join(args.datadir,args.diffplot_name2)
    if args.motion_outfile_prefix is None:
        motion_outfile_prefix = save_filename_prefix1 + "_" + dt_match2 + "motion"
    else:
        motion_outfile_prefix = os.path.join(args.datadir,args.motion_outfile_prefix)
    if args.gifname is None:
        gifname = save_filename_prefix1 + "_" + dt_match2 + "_anime.gif"
    else:
        gifname = os.path.join(args.datadir, args.gifname)



    plot_diff_labelled(args.rawfile1, args.rawfile2, diffcatalog_name1, diffcatalog_name2,
                       ind1=args.ind1, ind2=args.ind2,
                       motion_outfile_prefix=motion_outfile_prefix,
                       outfile1=diffplot_name1, outfile2=diffplot_name2,
                       cropxs=cropxs, cropys=cropys)

    grid2gif(diffplot_name1, diffplot_name2, gifname)