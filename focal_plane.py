#from astropy.table import Table
import sewpy
import argparse
import re

from matplotlib.patches import Ellipse

import matplotlib.pyplot as plt
import numpy as np

font = {'size': 14}
import matplotlib

matplotlib.rc('font', **font)

try:
    import cv2
    has_cv2 = True
except:
    print("Can't import cv2!!")
    has_cv2 = False



pix2mm = 0.48244
x_corners = np.array([1200,1210,1947,1935])
y_corners = np.array([1374, 635, 648, 1385])
center=np.array([np.mean(x_corners), np.mean(y_corners)])


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
    fitf.writeto(outfile, overwrite=overwrite)
    # return im


def plot_sew_cat(dst_trans, sew_out_trans,
                 brightestN=0,
                 xlim=None, ylim=None, outfile=None, show=False):
    # plt.figure()
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    ax_img = ax.imshow(dst_trans, cmap='gray')
    fig.colorbar(ax_img)

    # plt.plot(cat_test['X_IMAGE'], cat_test['Y_IMAGE'], color='r',  marker='o', ms=5, ls='')
    # plt.scatter(sew_out_trans['table']['X_IMAGE'],
    #            sew_out_trans['table']['Y_IMAGE'], s=40, facecolors='none', edgecolors='r')

    i = 0
    for row in sew_out_trans['table']:
        i += 1
        if brightestN > 0 and i > brightestN:
            break
        # print(row)
        e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]),
                    width=row['A_IMAGE'],
                    height=row['B_IMAGE'],
                    angle=row['THETA_IMAGE'],
                    linewidth=2, fill=False, )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('r')
        # e.set_facecolor('r')

    # plt.xlim(1350, 1800)
    # plt.ylim(1250,800)
    zoom = False
    if xlim is not None:
        plt.xlim(xlim)
        zoom = True
    if ylim is not None:
        plt.ylim(ylim)
        zoom = True

    if outfile is not None:
        fig.savefig(outfile)
    if show:
        plt.show()


def process_raw(rawfile, kernel_w = 3,
                DETECT_MINAREA = 30, THRESH = 5,
                sewpy_params=["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUX_RADIUS", "FLAGS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE"],
                cropxs=(1350, 1800), cropys=(1250, 800),
                savecatalog_name=None,
                savefits_name=None, overwrite_fits=True,
                saveplot_name=None):

    im_raw = read_raw(rawfile)

    if has_cv2:
        median = cv2.medianBlur(im_raw, kernel_w)
    else:
        print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
        median = im_raw

    im_std = np.std(median)
    print("Standard deviation of the image is {:.2f}".format(im_std))

    if savefits_name is None:
        savefits_name = rawfile[:-4]+".fits"
    elif savefits_name[-4:] != ("fits" or "fit"):
        print("Please provide a filename with fits or fit extension")
        exit(0)

    im2fits(median, savefits_name, overwrite=overwrite_fits)

    sew = sewpy.SEW(params=sewpy_params,
                    config={"DETECT_MINAREA": DETECT_MINAREA,
                            "BACK_SIZE":128 ,
                            "BACK_FILTERSIZE":3,
                            "DETECT_THRESH": THRESH, "ANALYSIS_THRESH": THRESH,
                            }
                    )

    sew_out = sew(savefits_name)
    sew_out['table'].sort('FLUX_ISO')
    sew_out['table'].reverse()
    n_sources = len(sew_out['table'])

    print("Found {} sources in file {}".format(n_sources, rawfile))

    plot_sew_cat(median, sew_out,
                 outfile=saveplot_name,
                 xlim=cropxs,
                 ylim=cropys)
    if savecatalog_name is not None:
        from astropy.io import ascii
        ascii.write(sew_out['table'], savecatalog_name, overwrite=True)
    return sew_out['table'], median



def naive_comparison(sew_out_table1, sew_out_table2, im1, im2,
                     min_dist=20,
                     outfile1=None, outfile2=None, verbose=False,
                     diffcat1="diff_cat1.txt", diffcat2="diff_cat2.txt",
                     cropxs=(1350, 1800), cropys=(1250, 800),
                     ):

    #xy_common=[]
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

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax_img = ax.imshow(im2, cmap='gray')
    fig.colorbar(ax_img)

    i2 = -1
    for row in sew_out_table2:
        x2_ = row['X_IMAGE']
        y2_ = row['Y_IMAGE']
        f2_ = row['FLUX_ISO']
        xy2_ = np.array([x2_, y2_])
        i2+=1
        i1 = -1
        for row1 in sew_out_table1:
            i1+=1
            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            xy1_ = np.array([x1_, y1_])
            dist_ = np.linalg.norm(xy1_-xy2_)
            if dist_ <= min_dist:
                #xy_common.append((xy1_+xy2_)/2.)
                commond_ind1.append(i1)
                commond_ind2.append(i2)
                e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]),
                            width=row['A_IMAGE'],
                            height=row['B_IMAGE'],
                            angle=row['THETA_IMAGE'],
                            linewidth=2, fill=False,)
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('r')
                #e.set_facecolor('r')
                try:
                    diff_ind1.remove(i1)
                except:
                    #if verbose:
                    #    print("Double Match 1...... !!!!!")
                    #    print(diff_ind1)
                    #else:
                    pass
                try:
                    diff_ind2.remove(i2)
                except:
                    #if verbose:
                    #    print("Double Match 2...... !!!!!")
                    #    print(diff_ind2)
                    #else:
                    pass


    i2 = -1
    for row1 in sew_out_table2:
            i2+=1
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

            #xy_diff.append((xy1_+xy2_)/2.)

            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                            width=row1['A_IMAGE'],
                            height=row1['B_IMAGE'],
                            angle=row1['THETA_IMAGE'],
                            linewidth=2, fill=False,)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')


    #xy_common=np.array(xy_common)
    #xy_diff=np.array(xy_diff)
    #print(diff_ind1, diff_ind2)
    #xy_common.shape #, xy_diff.shape
    if verbose:
        print("Common indices")
        print(commond_ind1, commond_ind2)

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)
    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners,
                y_corners, s=20, facecolors='none', edgecolors='m')

    if outfile2 is not None:
        plt.savefig(outfile2)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})


    ax_img = ax.imshow(im1, cmap='gray')
    fig.colorbar(ax_img)

    i1 = -1
    for row1 in sew_out_table1:
            i1+=1
            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            xy1_ = np.array([x1_, y1_])

            dist_ = np.linalg.norm(xy1_-xy2_)
            if i1 in commond_ind1:
                e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                            width=row1['A_IMAGE'],
                            height=row1['B_IMAGE'],
                            angle=row1['THETA_IMAGE'],
                            linewidth=2, fill=False,)
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('r')


    i1 = -1
    for row1 in sew_out_table1:
            i1+=1
            if i1 not in diff_ind1:
                continue

            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            xy1_ = np.array([x1_, y1_])
            #xy_diff.append((xy1_+xy2_)/2.)
            if verbose:
                print("==== New in catalog 1 ====")
                print(xy1_)
            with open(diffcat1, 'a') as diffcat1_io:
                for c_ in sew_out_table1.colnames:
                    diffcat1_io.write(str(row1[c_]))
                    diffcat1_io.write(" ")
                diffcat1_io.write("\n")
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                            width=row1['A_IMAGE'],
                            height=row1['B_IMAGE'],
                            angle=row1['THETA_IMAGE'],
                            linewidth=2, fill=False,)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')

    if cropxs is not None:
        plt.xlim(cropxs)
    if cropys is not None:
        plt.ylim(cropys)

    #mark center and corners:
    plt.plot([center[0]], [center[1]], 'g+')
    plt.scatter(x_corners,
                y_corners, s=20, facecolors='none', edgecolors='m')

    if outfile1 is not None:
        plt.savefig(outfile1)


def calc_dist(cat1, cat2, n1, n2, pix2mm = 0.48244):
    dx = (cat2.loc[n2]['X_IMAGE'] - cat1.loc[n1]['X_IMAGE'])
    dy = (cat2.loc[n2]['Y_IMAGE'] - cat1.loc[n1]['Y_IMAGE'])
    dist_pix = np.sqrt(dx**2 + dy**2)
    dist_mm = dist_pix * pix2mm
    print("centroid moved in x {:.4f} pix and in y {:.4f} pix".format(dx, dy))
    print("centroid moved by distance {:.4f} pix = {:.4f} mm".format(dist_pix, dist_mm))

    return dx, dy, dist_pix


def plot_diff_labelled(rawf1, rawf2, cat1, cat2,
                       ind1=None, ind2=None,
                       motion_outfile_prefix="motion_output",
                       outfile1="new_label1.pdf", outfile2="new_label2.pdf",
                       cropxs=(1350, 1800), cropys=(1250, 800),):
    import pandas as pd

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

    ax_img = ax.imshow(im1, cmap='gray')
    fig.colorbar(ax_img)

    if ind1 is not None:
        motion_outfile = motion_outfile_prefix+"ind"+str(ind1)+"_ind"+str(ind2)+".txt"

        row1 = cat1.loc[ind1]
        with open(motion_outfile, 'w') as io_:
            io_.write("# center (hard coded): ({}, {})\n".format(center[0], center[1]))
            io_.write(" ".join(cat1.columns))
            io_.write("\n")
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    width=row1['A_IMAGE'],
                    height=row1['B_IMAGE'],
                    angle=row1['THETA_IMAGE'],
                    linewidth=2, fill=False, )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('g')
        ax.annotate(str(ind1), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])),
                    color='g',
                    arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=2, headlength=4, width=1),
                    )
    else:
        for i, row1 in cat1.iterrows():
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE'],
                        height=row1['B_IMAGE'],
                        angle=row1['THETA_IMAGE'],
                        linewidth=2, fill=False, )
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')
            ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])),
                        color='g',
                        arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=2, headlength=4, width=1),
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

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    ax_img = ax.imshow(im2, cmap='gray')
    fig.colorbar(ax_img)

    if ind2 is not None:
        motion_outfile = motion_outfile_prefix + "ind" + str(ind1) + "_ind" + str(ind2) + ".txt"
        row1 = cat2.loc[ind2]
        with open(motion_outfile, 'a') as io_:
            for c_ in cat1.columns:
                io_.write(str(row1[c_]))
                io_.write(" ")
            io_.write("\n")

        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    width=row1['A_IMAGE'],
                    height=row1['B_IMAGE'],
                    angle=row1['THETA_IMAGE'],
                    linewidth=2, fill=False, )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('y')
        ax.annotate(str(ind2), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])),
                    color='y',
                    arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=2, headlength=4, width=1),
                    )
    else:
        for i, row1 in cat2.iterrows():
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE'],
                        height=row1['B_IMAGE'],
                        angle=row1['THETA_IMAGE'],
                        linewidth=2, fill=False, )
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')
            ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])),
                        color='y',
                        arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=2, headlength=4, width=1),
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


def get_datetime_rawname(raw_name):
    pattern = r'\b\w{1,4}-\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2}'
    match = re.search(pattern, raw_name)
    dt_match = match.group()  # raw_name[:match.start()]
    dt_match = "_".join(dt_match.split('-'))
    dt_match = "_".join(dt_match.split(':'))
    return dt_match


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare two raw images of stars at the focal plane')

    parser.add_argument('rawfile1',type=str)
    parser.add_argument('rawfile2',type=str)

    parser.add_argument('--DETECT_MINAREA', type=int, default=30, help="+++ Important parameter +++: "
                                                                       "Config param for sextractor, our default is 30.")
    parser.add_argument('--THRESH', type=int, default=6, help="+++ Important parameter +++: "
                                                              "Config param for sextractor, our default is 6.")

    parser.add_argument('--kernel_w', type=int, default=3, help="If you have cv2, this is the median blurring kernel width"
                                                                "our default is 3 (for a 3x3 kernel).")

    parser.add_argument('--min_dist', type=float, default=20, help="+++ Important parameter +++: "
                                                                   "Minimum distance we use to conclude a centroid is common in both images"
                                                                   "our default is 20 pixels.")

    parser.add_argument('--save_filename_prefix1', default=None,
                        help="File name prefix of the output files, "
                             "this will automatically populate savefits_name1, saveplot_name1, savecatalog_name1, and diffcatalog_name1"
                             "default is None.")

    parser.add_argument('--save_filename_prefix2', default=None,
                        help="File name prefix of the output files, "
                             "this will automatically populate savefits_name2, saveplot_name2, savecatalog_name2, and diffcatalog_name2"
                             "default is None.")

    parser.add_argument('--savefits_name1',default=None,
                        help="File name of the fits for the first image if you want to choose, "
                             "default has the same name as raw.")
    parser.add_argument('--saveplot_name1',default=None,
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


    parser.add_argument('--cropx1',
                        #default=1650,
                        default=1170,
                        #default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")
    parser.add_argument('--cropx2',
                        #default=2100,
                        default=1970,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")

    parser.add_argument('--cropy1',
                        default=1410,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")
    parser.add_argument('--cropy2',
                        default=610,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")

    parser.add_argument('-o', '--motion_outfile_prefix', dest="motion_outfile_prefix", default="motion_output",
                        help="File name prefix of the output catalog to calculate motion.")

    parser.add_argument('--nozoom', action='store_true', help="Do not zoom/crop the image. ")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.nozoom:
        cropxs=None
        cropys=None
    else:
        cropxs = (args.cropx1, args.cropx2)
        cropys = (args.cropy1, args.cropy2)

    if args.save_filename_prefix1 is not None:
        savefits_name1 = args.save_filename_prefix1+'_im1.fits'
        savecatalog_name1 = args.save_filename_prefix1+'_cat1.txt'
        saveplot_name1 = args.save_filename_prefix1+'_cat1.pdf'
        diffcatalog_name1 = args.save_filename_prefix1 + "_diff_cat1.txt"
        diffplot_name1 = args.save_filename_prefix1 + "_diff_cat1.pdf"
        motion_outfile_prefix =  args.save_filename_prefix1
    elif args.savefits_name1 is None or args.savecatalog_name1 is None or args.diffcatalog_name1 is None or args.diffplot_name1 is None:
        dt_match = get_datetime_rawname(args.rawfile1)
        print("Using default output file names with date {}".format(dt_match))
        save_filename_prefix1 = "res_focal_plane_"+dt_match
        savefits_name1 = save_filename_prefix1+'_im1.fits'
        savecatalog_name1 = save_filename_prefix1+'_cat1.txt'
        saveplot_name1 = save_filename_prefix1+'_cat1.pdf'
        diffcatalog_name1 = save_filename_prefix1 + "_diff_cat1.txt"
        diffplot_name1 = save_filename_prefix1 + "_diff_cat1.pdf"
        motion_outfile_prefix = save_filename_prefix1
    else:
        savefits_name1 = args.savefits_name1
        savecatalog_name1 = args.savecatalog_name1
        saveplot_name1 = args.saveplot_name1
        diffcatalog_name1 = args.diffcatalog_name1
        diffplot_name1 = args.diffplot_name1
        motion_outfile_prefix = args.motion_outfile_prefix
    if args.save_filename_prefix2 is not None:
        savefits_name2 = args.save_filename_prefix2 + '_im2.fits'
        savecatalog_name2 = args.save_filename_prefix2 + '_cat2.txt'
        saveplot_name2 = args.save_filename_prefix2 + '_cat2.pdf'
        diffcatalog_name2 = args.save_filename_prefix2 + "_diff_cat2.txt"
        diffplot_name2 = args.save_filename_prefix2 + "_diff_cat2.pdf"
        motion_outfile_prefix =  motion_outfile_prefix + "_" + args.save_filename_prefix2 + "_motion.txt"
    elif args.savefits_name2 is None or args.savecatalog_name2 is None or args.diffcatalog_name2 is None or args.diffplot_name2 is None:
        dt_match = get_datetime_rawname(args.rawfile2)
        print("Using default output file names with date {}".format(dt_match))
        save_filename_prefix2 = "res_focal_plane_" + dt_match
        savefits_name2 = save_filename_prefix2 + '_im2.fits'
        savecatalog_name2 = save_filename_prefix2 + '_cat2.txt'
        saveplot_name2 = save_filename_prefix2 + '_cat2.pdf'
        diffcatalog_name2 = save_filename_prefix2 + "_diff_cat2.txt"
        diffplot_name2 = save_filename_prefix2 + "_diff_cat2.pdf"
        motion_outfile_prefix = motion_outfile_prefix + "_" + dt_match + "motion"
    else:
        savefits_name2 = args.savefits_name2
        savecatalog_name2 = args.savecatalog_name2
        saveplot_name2 = args.saveplot_name2
        diffcatalog_name2 = args.diffcatalog_name2
        diffplot_name2 = args.diffplot_name2
        motion_outfile_prefix = args.motion_outfile_prefix

    sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                sewpy_params=["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUX_RADIUS", "FLAGS", "A_IMAGE", "B_IMAGE",
                              "THETA_IMAGE"],
                cropxs=cropxs, cropys=cropys,
                savefits_name=savefits_name1, overwrite_fits=True,
                saveplot_name=None, savecatalog_name=savecatalog_name1
                                          )

    sew_out_table2, im_med2 = process_raw(args.rawfile2, kernel_w=args.kernel_w,
                DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                sewpy_params=["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUX_RADIUS", "FLAGS", "A_IMAGE", "B_IMAGE",
                              "THETA_IMAGE"],
                cropxs=cropxs, cropys=cropys,
                savefits_name=savefits_name2, overwrite_fits=True,
                saveplot_name=None, savecatalog_name=savecatalog_name2
                                          )

    naive_comparison(sew_out_table1, sew_out_table2, im_med1, im_med2,
                     min_dist=args.min_dist,
                     diffcat1=diffcatalog_name1, diffcat2=diffcatalog_name2,
                     outfile1=saveplot_name1, outfile2=saveplot_name2,
                     cropxs=cropxs, cropys=cropys,
                     verbose=args.verbose
                     )

    plot_diff_labelled(args.rawfile1, args.rawfile2, diffcatalog_name1, diffcatalog_name2,
                       ind1=None, ind2=None,
                       motion_outfile_prefix=motion_outfile_prefix,
                       outfile1=diffplot_name1, outfile2=diffplot_name2,
                       cropxs=cropxs, cropys=cropys)


