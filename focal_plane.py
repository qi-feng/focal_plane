#from astropy.table import Table
import sewpy
import argparse
import re
import os
import pandas as pd
import subprocess


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

pattern_label_x_min = 1500
pattern_label_x_max = 1900
pattern_label_y_min = 1500
pattern_label_y_max = 1850
center_pattern = np.array([(pattern_label_x_min+pattern_label_x_max)/2., (pattern_label_y_min+pattern_label_y_max)/2.])

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
                 xlim=None, ylim=None, outfile=None, show=False, vmax=None):
    # plt.figure()
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    if vmax is not None:
        ax_img = ax.imshow(dst_trans, vmax=vmax, cmap='gray')
    else:
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


        kr = row['KRON_RADIUS']
        e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]),
                    width=row['A_IMAGE']*kr,
                    height=row['B_IMAGE']*kr,
                    angle=row['THETA_IMAGE'],
                    linewidth=1, fill=False, alpha=0.9)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('r')
        ax.add_artist(e)

        #if row['X_IMAGE'] <= 2100 and row['X_IMAGE'] >= 1700 and row['Y_IMAGE'] >= 880 and row['Y_IMAGE'] <= 1250:
        if row['X_IMAGE'] <= pattern_label_x_max and row['X_IMAGE'] >= pattern_label_x_min and row[
            'Y_IMAGE'] >= pattern_label_y_min and row[
            'Y_IMAGE'] <= pattern_label_y_max:
            #print("Yo")
            #print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
            ax.annotate(int(row['ID']), xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data',
                        #xytext=(np.array([row['X_IMAGE'] - 40, row['Y_IMAGE'] - 40])), # for orig lens
                        xytext=(np.array([row['X_IMAGE'] + row['X_IMAGE'] - center_pattern[0],
                                          row['Y_IMAGE'] + row['Y_IMAGE'] - center_pattern[1]])),  # for orig lens
                        #xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                    color='c', alpha=0.8,
                    arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=4, width=0.5,
                                    alpha=0.7),
                    )
            e = Ellipse(xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]),
                        width=row['A_IMAGE'] * kr,
                        height=row['B_IMAGE'] * kr,
                        angle=row['THETA_IMAGE'],
                        linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_color('c')
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

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile)
    if show:
        plt.show()


def process_raw(rawfile, kernel_w = 3,
                DETECT_MINAREA = 30, THRESH = 5,
                clean=True,
                sewpy_params=["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "KRON_RADIUS", "FLUX_RADIUS", "FLAGS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE"],
                cropxs=(1350, 1800), cropys=(1250, 800),
                savecatalog_name=None,
                savefits_name=None, overwrite_fits=True,
                saveplot_name=None):
    from astropy.table import Column

    im_raw = read_raw(rawfile)

    if has_cv2:
        median = cv2.medianBlur(im_raw, kernel_w)
    else:
        print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
        median = im_raw

    im_std = np.std(median)
    print("Standard deviation of the image is {:.2f}".format(im_std))

    # plt.xlim(1050, 2592)
    # plt.ylim(1850,250)
    # plt.imshow(median_screen[250:1850, 1050:2592], cmap='gray')
    max_pixel_crop = np.max(median[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    print("Brightest pixel in the zoomed area reaches {}".format(max_pixel_crop))
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
                            "DEBLEND_MINCONT": 0.02,
                            }
                    )

    sew_out = sew(savefits_name)
    sew_out['table'].sort('FLUX_ISO')
    sew_out['table'].reverse()
    sew_out['table']['FLUX_AREA'] = np.pi * sew_out['table']['KRON_RADIUS'] * sew_out['table']['KRON_RADIUS'] * \
                                    sew_out['table']['A_IMAGE'] * sew_out['table']['A_IMAGE']
    if clean:
        sew_out['table'] = sew_out['table'][sew_out['table']['FLAGS'] <= 16]
        #sew_out['table'] = sew_out['table'][(sew_out['table']['FLUX_ISO'] / sew_out['table']['FLUX_AREA']) > 0.3]
    n_sources = len(sew_out['table'])
    ID_ = Column(range(n_sources), name='ID')
    sew_out['table'].add_column(ID_, index=0)
    sew_out['table'].add_index('ID')

    print("Found {} sources in file {}".format(n_sources, rawfile))

    plot_sew_cat(median, sew_out,
                 outfile=saveplot_name,
                 xlim=cropxs,
                 ylim=cropys, vmax=max_pixel_crop)
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

    max_pixel_crop2 = np.max(im2[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    #print("Brightest pixel in the zoomed area in image 2 reaches {}".format(max_pixel_crop2))
    #if max_pixel_crop2 == 255:
    #    print("Image 2 is saturated. ")

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax_img = ax.imshow(im2, cmap='gray', vmax=max_pixel_crop2)
    fig.colorbar(ax_img)

    i2 = -1

    for row in sew_out_table2:
        x2_ = row['X_IMAGE']
        y2_ = row['Y_IMAGE']
        f2_ = row['FLUX_ISO']
        kr = row['KRON_RADIUS']
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
                            width=row['A_IMAGE']*kr,
                            height=row['B_IMAGE']*kr,
                            angle=row['THETA_IMAGE'],
                            linewidth=1, fill=False, alpha=0.9)
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('r')

                #if row['X_IMAGE'] <= 2100 and row['X_IMAGE'] >= 1700 and row['Y_IMAGE'] >= 880 and row[
                #    'Y_IMAGE'] <= 1250:
                if row['X_IMAGE'] <= pattern_label_x_max and row['X_IMAGE'] >= pattern_label_x_min and row[
                    'Y_IMAGE'] >= pattern_label_y_min and row[
                    'Y_IMAGE'] <= pattern_label_y_max:
                    #print("Yo")
                    #print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                    ax.annotate(int(row['ID']), xy=np.array([row['X_IMAGE'], row['Y_IMAGE']]), size=8, xycoords='data',
                                xytext=(np.array([row['X_IMAGE'] + row['X_IMAGE'] - center_pattern[0],
                                                  row['Y_IMAGE'] + row['Y_IMAGE'] - center_pattern[1]])),  # for orig lens
                                #xytext=(np.array([row['X_IMAGE'] - 80, row['Y_IMAGE'] - 80])),  # for new lens
                                color='c', alpha=0.8,
                                arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                                width=0.5,
                                                alpha=0.7),
                                )

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
            kr1 = row1['KRON_RADIUS']
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
                            width=row1['A_IMAGE']*kr1,
                            height=row1['B_IMAGE']*kr1,
                            angle=row1['THETA_IMAGE'],
                            linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')

            #if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
            #    'Y_IMAGE'] <= 1250:
            if row1['X_IMAGE'] <= pattern_label_x_max and row1['X_IMAGE'] >= pattern_label_x_min and row1[
                    'Y_IMAGE'] >= pattern_label_y_min and row1[
                    'Y_IMAGE'] <= pattern_label_y_max:
                # print("Yo")
                # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8, xycoords='data',
                            xytext=(np.array([ row1['X_IMAGE'] + row1['X_IMAGE'] - center_pattern[0] ,
                                               row1['Y_IMAGE'] + row1['Y_IMAGE'] - center_pattern[1] ])),  # for orig lens
                            #xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                            color='c', alpha=0.8,
                            arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                            width=0.5,
                                            alpha=0.7),
                            )

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

    max_pixel_crop1 = np.max(im1[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
    #print("Brightest pixel in the zoomed area in image 1 reaches {}".format(max_pixel_crop1))
    #if max_pixel_crop1 == 255:
    #    print("Image 1 is saturated. ")

    ax_img = ax.imshow(im1, cmap='gray', vmax=max_pixel_crop1)
    fig.colorbar(ax_img)

    i1 = -1
    for row1 in sew_out_table1:
            i1+=1
            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            kr1 = row1['KRON_RADIUS']
            xy1_ = np.array([x1_, y1_])

            dist_ = np.linalg.norm(xy1_-xy2_)
            if i1 in commond_ind1:
                e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                            width=row1['A_IMAGE']*kr1,
                            height=row1['B_IMAGE']*kr1,
                            angle=row1['THETA_IMAGE'],
                            linewidth=1, fill=False, alpha=0.9)
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.8)
                e.set_color('r')

                #if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
                #    'Y_IMAGE'] <= 1250:
                if row1['X_IMAGE'] <= pattern_label_x_max and row1['X_IMAGE'] >= pattern_label_x_min and row1[
                    'Y_IMAGE'] >= pattern_label_y_min and row1[
                    'Y_IMAGE'] <= pattern_label_y_max:
                    # print("Yo")
                    # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                    ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                                xycoords='data',
                                xytext=(np.array([row1['X_IMAGE'] + row1['X_IMAGE'] - center_pattern[0],
                                                  row1['Y_IMAGE'] + row1['Y_IMAGE'] - center_pattern[1]])),  # for orig lens

                                #xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                                color='c', alpha=0.8,
                                arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                                width=0.5,
                                                alpha=0.7),
                                )

    i1 = -1
    for row1 in sew_out_table1:
            i1+=1
            if i1 not in diff_ind1:
                continue

            x1_ = row1['X_IMAGE']
            y1_ = row1['Y_IMAGE']
            f1_ = row1['FLUX_ISO']
            kr1 = row1['KRON_RADIUS']
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
                            width=row1['A_IMAGE']*kr1,
                            height=row1['B_IMAGE']*kr1,
                            angle=row1['THETA_IMAGE'],
                            linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')
            #if row1['X_IMAGE'] <= 2100 and row1['X_IMAGE'] >= 1700 and row1['Y_IMAGE'] >= 880 and row1[
            #    'Y_IMAGE'] <= 1250:
            if row1['X_IMAGE'] <= pattern_label_x_max and row1['X_IMAGE'] >= pattern_label_x_min and row1[
                'Y_IMAGE'] >= pattern_label_y_min and row1[
                'Y_IMAGE'] <= pattern_label_y_max:
                # print("Yo")
                # print(int(row['ID']), row['X_IMAGE'], row['Y_IMAGE'])
                ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8, xycoords='data',
                            xytext=(np.array([ row1['X_IMAGE'] + row1['X_IMAGE'] - center_pattern[0] ,
                                               row1['Y_IMAGE'] + row1['Y_IMAGE'] - center_pattern[1] ])),  # for orig lens
                            #xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])),  # for new lens
                            color='c', alpha=0.8,
                            arrowprops=dict(facecolor='c', edgecolor='c', shrink=0.05, headwidth=1, headlength=2,
                                            width=0.5,
                                            alpha=0.7),
                            )

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
        sat_pix = np.sum(im1 == 255)
        percent_satur = 100.0 * sat_pix / (cropys[0] - cropys[1]) / (cropxs[1] - cropxs[0])
        print("{:.3f}% of the pixels (= {} pixels) in zoomed area of image 1 is saturated. ".format(percent_satur, sat_pix))

    ax_img = ax.imshow(im1, cmap='gray', vmax=max_pixel_crop1)
    #ax_img = ax.imshow(im1, cmap='gray')
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
        kr = row1['KRON_RADIUS']
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    width=row1['A_IMAGE']*kr,
                    height=row1['B_IMAGE']*kr,
                    angle=row1['THETA_IMAGE'],
                    linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('g')
        #ax.annotate(str(ind1), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
        ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                    #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                    xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])), # for new lens
                    color='g', alpha=0.8,
                    arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=1, headlength=4, width=0.5, alpha=0.7),
                    )
    else:
        for i, row1 in cat1.iterrows():
            kr = row1['KRON_RADIUS']
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE']*kr,
                        height=row1['B_IMAGE']*kr,
                        angle=row1['THETA_IMAGE'],
                        linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('g')
            #ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),size=8,
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                        #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), #for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])), # for new lens
                        color='g', alpha=0.8,
                        arrowprops=dict(facecolor='g', edgecolor='g', shrink=0.05, headwidth=1, headlength=4, width=0.5, alpha=0.7),
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
        "{:.3f}% of the pixels (= {} pixels) in zoomed area of image 2 is saturated. ".format(percent_satur, sat_pix))


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
        kr = row1['KRON_RADIUS']
        e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                    width=row1['A_IMAGE']*kr,
                    height=row1['B_IMAGE']*kr,
                    angle=row1['THETA_IMAGE'],
                    linewidth=1, fill=False, alpha=0.9)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.8)
        e.set_color('y')
        #ax.annotate(str(ind2), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
        ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                    #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                    xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])), # for new lens
                    color='y',alpha=0.8,
                    arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=1, headlength=4, width=0.5, alpha=0.7),
                    )
    else:
        for i, row1 in cat2.iterrows():
            kr = row1['KRON_RADIUS']
            e = Ellipse(xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]),
                        width=row1['A_IMAGE']*kr,
                        height=row1['B_IMAGE']*kr,
                        angle=row1['THETA_IMAGE'],
                        linewidth=1, fill=False, alpha=0.9)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.8)
            e.set_color('y')
            #ax.annotate(str(i), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
            ax.annotate(int(row1['ID']), xy=np.array([row1['X_IMAGE'], row1['Y_IMAGE']]), size=8,
                        #xytext=(np.array([row1['X_IMAGE'] - 40, row1['Y_IMAGE'] - 40])), # for orig lens
                        xytext=(np.array([row1['X_IMAGE'] - 80, row1['Y_IMAGE'] - 80])), # for new lens
                        color='y', alpha=0.8,
                        arrowprops=dict(facecolor='y', edgecolor='y', shrink=0.05, headwidth=1, headlength=4, width=0.5,alpha=0.7 ),
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


def yes_or_no(question):
    reply = str(raw_input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter y or n")


def grid2gif(im1, im2, output_gif):
    str1 = 'convert -delay 50 -loop 0 -quality 100 -density 144 ' + im1 +' ' + im2  + ' ' + output_gif
    subprocess.call(str1, shell=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare two raw images of stars at the focal plane')

    parser.add_argument('rawfile1',type=str)
    parser.add_argument('rawfile2',type=str, default=None, nargs='?')

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
    parser.add_argument('--datadir', default="data",
                        help="Folder to save all output files. Default is ./data (ignored by git)")
    parser.add_argument('--gifname', default=None, #default="compare.gif",
                        help="File name to save gif animation. ")

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

    parser.add_argument('-o', '--motion_outfile_prefix', dest="motion_outfile_prefix", default="motion_output",
                        help="File name prefix of the output catalog to calculate motion.")

    parser.add_argument('--nozoom', action='store_true', help="Do not zoom/crop the image. ")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--clean', action='store_true', help="Whether or not to delete centroid with flag > 16.")
    parser.add_argument('-s', '--single', action='store_true', help="Only analyze a single image.")

    args = parser.parse_args()

    if args.nozoom:
        cropxs=None
        cropys=None
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


    if args.save_filename_prefix1 is not None:
        savefits_name1 = os.path.join(args.datadir, args.save_filename_prefix1+'_im1.fits')
        savecatalog_name1 = os.path.join(args.datadir,args.save_filename_prefix1+'_cat1.txt')
        saveplot_name1 = os.path.join(args.datadir,args.save_filename_prefix1+'_cat1.pdf')
        diffcatalog_name1 = os.path.join(args.datadir,args.save_filename_prefix1 + "_diff_cat1.txt")
        diffplot_name1 = os.path.join(args.datadir,args.save_filename_prefix1 + "_diff_cat1.pdf")
        motion_outfile_prefix =  os.path.join(args.datadir,args.save_filename_prefix1)
    elif args.savefits_name1 is None or args.savecatalog_name1 is None or args.diffcatalog_name1 is None or args.diffplot_name1 is None:
        dt_match = get_datetime_rawname(args.rawfile1)
        print("Using default output file names with date {}".format(dt_match))
        save_filename_prefix1 = os.path.join(args.datadir,"res_focal_plane_"+dt_match)
        savefits_name1 = save_filename_prefix1+'_im1.fits'
        savecatalog_name1 = save_filename_prefix1+'_cat1.txt'
        saveplot_name1 = save_filename_prefix1+'_cat1.pdf'
        diffcatalog_name1 = save_filename_prefix1 + "_diff_cat1.txt"
        diffplot_name1 = save_filename_prefix1 + "_diff_cat1.pdf"
        motion_outfile_prefix = save_filename_prefix1
    else:
        savefits_name1 = os.path.join(args.datadir,args.savefits_name1)
        savecatalog_name1 = os.path.join(args.datadir,args.savecatalog_name1)
        saveplot_name1 = os.path.join(args.datadir,args.saveplot_name1)
        diffcatalog_name1 = os.path.join(args.datadir,args.diffcatalog_name1)
        diffplot_name1 = os.path.join(args.datadir,args.diffplot_name1)
        motion_outfile_prefix = os.path.join(args.datadir,args.motion_outfile_prefix)
    if args.rawfile2 is not None:
        if args.save_filename_prefix2 is not None:
            savefits_name2 = os.path.join(args.datadir,args.save_filename_prefix2 + '_im2.fits')
            savecatalog_name2 = os.path.join(args.datadir,args.save_filename_prefix2 + '_cat2.txt')
            saveplot_name2 = os.path.join(args.datadir,args.save_filename_prefix2 + '_cat2.pdf')
            diffcatalog_name2 = os.path.join(args.datadir,args.save_filename_prefix2 + "_diff_cat2.txt")
            diffplot_name2 = os.path.join(args.datadir,args.save_filename_prefix2 + "_diff_cat2.pdf")
            motion_outfile_prefix =  motion_outfile_prefix + "_" + args.save_filename_prefix2 + "_motion.txt"
            gifname = motion_outfile_prefix + "_" + args.save_filename_prefix2 + "_anime.gif"
        elif args.savefits_name2 is None or args.savecatalog_name2 is None or args.diffcatalog_name2 is None or args.diffplot_name2 is None:
            dt_match = get_datetime_rawname(args.rawfile2)
            print("Using default output file names with date {}".format(dt_match))
            save_filename_prefix2 = os.path.join(args.datadir,"res_focal_plane_" + dt_match)
            savefits_name2 = save_filename_prefix2 + '_im2.fits'
            savecatalog_name2 = save_filename_prefix2 + '_cat2.txt'
            saveplot_name2 = save_filename_prefix2 + '_cat2.pdf'
            diffcatalog_name2 = save_filename_prefix2 + "_diff_cat2.txt"
            diffplot_name2 = save_filename_prefix2 + "_diff_cat2.pdf"
            gifname = motion_outfile_prefix + "_" + dt_match + "_anime.gif"
            motion_outfile_prefix = motion_outfile_prefix + "_" + dt_match + "motion"
        else:
            savefits_name2 = os.path.join(args.datadir,args.savefits_name2)
            savecatalog_name2 = os.path.join(args.datadir,args.savecatalog_name2)
            saveplot_name2 = os.path.join(args.datadir,args.saveplot_name2)
            diffcatalog_name2 = os.path.join(args.datadir,args.diffcatalog_name2)
            diffplot_name2 = os.path.join(args.datadir,args.diffplot_name2)
            motion_outfile_prefix = args.motion_outfile_prefix
            gifname = os.path.join(args.datadir, args.gifname)
        if args.gifname is not None:
            gifname = os.path.join(args.datadir, args.gifname)

    sew_params = ["X_IMAGE", "Y_IMAGE", "FLUX_ISO", "FLUX_RADIUS", "FLAGS", "KRON_RADIUS", "A_IMAGE", "B_IMAGE",
                              "THETA_IMAGE"]

    if args.single or args.rawfile2 is None:
        sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                              DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                                              sewpy_params=sew_params,
                                              cropxs=cropxs, cropys=cropys,
                                              clean=args.clean,
                                              savefits_name=savefits_name1, overwrite_fits=True,
                                              saveplot_name=saveplot_name1, savecatalog_name=savecatalog_name1,
                                              )
        print("Processing single image. Done.")
        exit(0)
    else:
        sew_out_table1, im_med1 = process_raw(args.rawfile1, kernel_w=args.kernel_w,
                                              DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                                              sewpy_params=sew_params,
                                              cropxs=cropxs, cropys=cropys,
                                              clean=args.clean,
                                              savefits_name=savefits_name1, overwrite_fits=True,
                                              saveplot_name=None, savecatalog_name=savecatalog_name1
                                              )
        sew_out_table2, im_med2 = process_raw(args.rawfile2, kernel_w=args.kernel_w,
                    DETECT_MINAREA=args.DETECT_MINAREA, THRESH=args.THRESH,
                    sewpy_params=sew_params,
                    cropxs=cropxs, cropys=cropys,
                    clean=args.clean,
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

        grid2gif(diffplot_name1, diffplot_name2, gifname)