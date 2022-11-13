import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sewpy
from matplotlib.patches import Ellipse
import argparse
import re


font = {'size': 14}
import matplotlib
# from sklearn import cluster
from sklearn.cluster import KMeans

# edge
from scipy import ndimage as ndi
from skimage import feature

# noise
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage import color

import scipy as sp
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse, Rectangle

try:
    import cv2
except:
    print("Can't import cv2!!")

matplotlib.rc('font', **font)


#PIX2MM = 0.241
PIX2MM = 0.482
#PIX2MM = 0.477

MM2ARCMIN = 1. / 1.625  # ???


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


def chisq(y_vals, y_expected, y_errs=1, num_params=1):
    # returns chi2, dof, red_chi2
    #  for reduced chisq test, under the assumption of Poisson counting
    #  we have lnL = const - (1/2.)*chi2
    if y_vals.shape[0] != y_expected.shape[0]:
        print("Inconsistent input sizes")
        return
    # z = (y_vals[i] - y_expected[i]) / y_errs[i]
    z = (y_vals - y_expected) / y_errs
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (y_vals.shape[0] - num_params)
    return chi2, (y_vals.shape[0] - num_params), chi2dof


# test baseline:

def gaussian_baseline(height, center_x, center_y, width_x, width_y, rotation, baseline):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)

    # center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    # center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        # xp = x * np.cos(rotation) - y * np.sin(rotation)
        # yp = x * np.sin(rotation) + y * np.cos(rotation)
        xp = (x - center_x) * np.cos(rotation) - (y - center_y) * np.sin(rotation)  # + center_x
        yp = (x - center_x) * np.sin(rotation) + (y - center_y) * np.cos(rotation)  # + center_y
        g = height * np.exp(
            -(((-xp) / width_x) ** 2 +
              ((-yp) / width_y) ** 2) / 2.) + baseline
        # -(((center_x-xp)/width_x)**2+
        #  ((center_y-yp)/width_y)**2)/2.)
        return g

    return rotgauss



def gaussian_0baseline(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)

    # center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    # center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        # xp = x * np.cos(rotation) - y * np.sin(rotation)
        # yp = x * np.sin(rotation) + y * np.cos(rotation)
        xp = (x - center_x) * np.cos(rotation) - (y - center_y) * np.sin(rotation)  # + center_x
        yp = (x - center_x) * np.sin(rotation) + (y - center_y) * np.cos(rotation)  # + center_y
        g = height * np.exp(
            -(((-xp) / width_x) ** 2 +
              ((-yp) / width_y) ** 2) / 2.)
        # -(((center_x-xp)/width_x)**2+
        #  ((center_y-yp)/width_y)**2)/2.)
        return g

    return rotgauss


def moments_baseline(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0, 0


def moments_0baseline(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0

def fitgaussian_baseline(data):
    """Returns (height, x, y, width_x, width_y, theta)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments_baseline(data)
    errorfunction = lambda p: np.ravel(gaussian_baseline(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = sp.optimize.leastsq(errorfunction, params)
    return p


def fitgaussian_0baseline(data):
    """Returns (height, x, y, width_x, width_y, theta)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments_0baseline(data)
    errorfunction = lambda p: np.ravel(gaussian_0baseline(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = sp.optimize.leastsq(errorfunction, params)
    return p

def fit_gaussian2d_baseline(data, outfile=None, df=None, log=False,
                            show_crop=0, PIX2MM=PIX2MM,
                            legend=False, draw_pixel=True):  # , amp=1, xc=0,yc=0,A=1,B=1,theta=0, offset=0):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # plt.matshow(data, cmap=plt.cm.gray)
    # plt.pcolor(data, cmap=plt.cm.gray)
    if log:
        from matplotlib.colors import LogNorm
        plt.imshow(data, cmap=plt.cm.gray, norm=LogNorm(vmin=0.01, vmax=np.max(data)))
    else:
        if show_crop:
            # plt.imshow(data[int(data.shape[1]/2-show_crop/2):int(data.shape[1]/2+show_crop/2),int(data.shape[0]/2-show_crop/2):int(data.shape[0]/2+show_crop/2)], cmap=plt.cm.gray)
            """
            plt.imshow(data[int(data.shape[1]/2-show_crop/2):int(data.shape[1]/2+show_crop/2),int(data.shape[0]/2-show_crop/2):int(data.shape[0]/2+show_crop/2)], cmap=plt.cm.gray,
            #plt.imshow(data, cmap=plt.cm.gray, 
                extent=[int(data.shape[1]/2-show_crop/2),int(data.shape[1]/2+show_crop/2),
                        int(data.shape[0]/2-show_crop/2),int(data.shape[0]/2+show_crop/2)]
                      )
            """
            plt.imshow(data[int(data.shape[0] / 2 - show_crop / 2):int(data.shape[0] / 2 + show_crop / 2),
                       int(data.shape[1] / 2 - show_crop / 2):int(data.shape[1] / 2 + show_crop / 2)], cmap=plt.cm.gray,
                       # plt.imshow(data, cmap=plt.cm.gray,
                       extent=[int(data.shape[0] / 2 - show_crop / 2), int(data.shape[0] / 2 + show_crop / 2),
                               int(data.shape[1] / 2 + show_crop / 2), int(data.shape[1] / 2 - show_crop / 2)]
                       )
            print(int(data.shape[0] / 2 - show_crop / 2), int(data.shape[0] / 2 + show_crop / 2),
                  int(data.shape[1] / 2 - show_crop / 2), int(data.shape[1] / 2 + show_crop / 2))
        else:
            plt.imshow(data, cmap=plt.cm.gray)

    plt.colorbar()
    params = fitgaussian_baseline(data)

    fit = gaussian_baseline(*params)

    # plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper, levels=[68, 90, 95])
    ax = plt.gca()
    (height, x, y, width_x, width_y, theta, baseline) = params
    # print(height, x, y, width_x, width_y, np.rad2deg(theta))
    print(height, x, y, width_x, width_y, theta, baseline)

    center_crop = data[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    print("Frac contained in central image pixel from data")
    print(center_crop.sum() / data.sum())
    center_crop = data[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    print("Frac contained in central trigger pixel from data")
    print(center_crop.sum() / data.sum())

    if legend:
        plt.text(0.95, 0.05, """
        x : %.1f
        y : %.1f
        $\sigma_x$ : %.1f pix \n = %.1f mm = %.1f \'
        $\sigma_y$ : %.1f pix \n = %.1f mm = %.1f \'""" % (
        y, x, width_y, width_y * PIX2MM, width_y * PIX2MM * MM2ARCMIN,
        width_x, width_x * PIX2MM, width_x * PIX2MM * MM2ARCMIN,),
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='y')
    # plt.plot([x,x], [y,y+width_y], ls='-', color='c')
    # plt.plot([x,x+width_x], [y,y], ls='-', color='c')
    # e = Ellipse(xy=np.array([x,y]), width=width_x*2,
    #            height=width_y*2, angle=-theta, linewidth=1, fill=False, alpha=0.9)
    e = Ellipse(xy=np.array([y, x]), width=width_y * 2,
                height=width_x * 2, angle=theta, linewidth=1, fill=False, alpha=0.9)

    ax.add_artist(e)
    e.set_color('c')

    sigma80perc = 1.794
    # e2 = Ellipse(xy=np.array([x,y]), width=width_x*4,
    #            height=width_y*4, angle=-theta, linewidth=1, fill=False, alpha=0.9)
    # below is 2 sigma
    # e2 = Ellipse(xy=np.array([y,x]), width=width_y*4,
    #            height=width_x*4, angle=theta, linewidth=1, fill=False, alpha=0.9)
    e2 = Ellipse(xy=np.array([y, x]), width=width_y * 2 * sigma80perc,
                 height=width_x * 2 * sigma80perc, angle=theta, linewidth=1, fill=False, alpha=0.9)

    ax.add_artist(e2)
    e2.set_color('c')

    sigma = 1.5
    sigma = 1.5
    ex = Ellipse(xy=np.array([y, x]), width=width_y * 2 * sigma,
                 height=width_x * 2 * sigma, angle=theta, linewidth=1, fill=False, alpha=0.9)

    ax.add_artist(ex)
    ex.set_color('c')

    if df is not None:
        kr = df['KRON_RADIUS']
        e3 = Ellipse(xy=np.array([y, x]), width=df['A_IMAGE'] * kr,
                     height=df['B_IMAGE'] * kr, angle=df['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        print("Sextractor diameter A_mm {} B_mm {}".format(df['A_IMAGE'] * kr * PIX2MM, df['B_IMAGE'] * kr * PIX2MM))
        # e3 = Ellipse(xy=np.array([y,x]), width=width_y*2,
        #        height=width_x*2, angle=theta, linewidth=1, fill=False, alpha=0.9)

        ax.add_artist(e3)
        e3.set_color('r')

    if draw_pixel:
        # Create a Rectangle patch
        # rect = Rectangle((70,70),25,25,#linewidth=1.5,
        # 3.7'
        # rect = Rectangle((5,65),25,25,#linewidth=1.5,
        #                 edgecolor='y',facecolor='none')
        # 4'
        # rect = Rectangle((4,62),6.53/PIX2MM,6.53/PIX2MM,#linewidth=1.5,
        pix_length = 6.53/PIX2MM
        rect = Rectangle((y - pix_length / 2, x - pix_length / 2), pix_length, pix_length,  # linewidth=1.5,
                         edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        pix_length = 6.53/PIX2MM * 2
        rect = Rectangle((y - pix_length / 2, x - pix_length / 2), pix_length, pix_length,  # linewidth=1.5,
                         edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        # plt.text(0.24, 0.02, "3.7\'",
        plt.text(0.54, 0.15, "8\'",
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='y')

    else:
        plt.plot([10, 10 + 3 / (PIX2MM * MM2ARCMIN)], [90, 90], 'y-')
        # plt.plot([10, 10+3/PIX2MM * (1./1.5)], [90,90], 'y-')
        plt.text(0.23, 0.02, "3\'",
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='y')

    plt.axis('off')
    plt.tight_layout()
    if outfile is not None:
        if outfile.split(".")[-1] == "png":
            plt.savefig(outfile, dpi=300, bbox_inches='tight',
                        pad_inches=0)
        else:
            plt.savefig(outfile, bbox_inches='tight',
                        pad_inches=0)

    print(r"""
        x : %.1f
        y : %.1f
        $\sigma_x$ : %.1f pix = %.1f mm = %.2f '
        $\sigma_y$ : %.1f pix = %.1f mm = %.2f '""" % (y, x, width_y, width_y * PIX2MM, width_y * PIX2MM * MM2ARCMIN,
                                                       width_x, width_x * PIX2MM, width_x * PIX2MM * MM2ARCMIN,))
    print_sigma = 1
    print("""
    %.1f $\sigma_x$ : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$ : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, print_sigma * width_y, print_sigma * width_y * PIX2MM, print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, print_sigma * width_x, print_sigma * width_x * PIX2MM, print_sigma * width_x * PIX2MM * MM2ARCMIN,))
    print("""
    %.1f $\sigma_x$x2 : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$x2 : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, 2 * print_sigma * width_y, 2 * print_sigma * width_y * PIX2MM,
    2 * print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, 2 * print_sigma * width_x, 2 * print_sigma * width_x * PIX2MM,
    2 * print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    print_sigma = 1.5
    print("""
    %.1f $\sigma_x$ : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$ : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, print_sigma * width_y, print_sigma * width_y * PIX2MM, print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, print_sigma * width_x, print_sigma * width_x * PIX2MM, print_sigma * width_x * PIX2MM * MM2ARCMIN,))
    print("""
    %.1f $\sigma_x$x2 : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$x2 : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, 2 * print_sigma * width_y, 2 * print_sigma * width_y * PIX2MM,
    2 * print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, 2 * print_sigma * width_x, 2 * print_sigma * width_x * PIX2MM,
    2 * print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    print_sigma = 1.8
    print("""
    %.1f $\sigma_x$ : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$ : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, print_sigma * width_y, print_sigma * width_y * PIX2MM, print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, print_sigma * width_x, print_sigma * width_x * PIX2MM, print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    print("""
    %.1f $\sigma_x$x2 : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$x2 : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, 2 * print_sigma * width_y, 2 * print_sigma * width_y * PIX2MM,
    2 * print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, 2 * print_sigma * width_x, 2 * print_sigma * width_x * PIX2MM,
    2 * print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    print_sigma = 2.146
    print("""
    %.1f $\sigma_x$ : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$ : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, print_sigma * width_y, print_sigma * width_y * PIX2MM, print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, print_sigma * width_x, print_sigma * width_x * PIX2MM, print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    print("""
    %.1f $\sigma_x$x2 : %.1f pix = %.2f mm = %.2f '
    %.1f $\sigma_y$x2 : %.1f pix = %.2f mm = %.2f '""" % (
    print_sigma, 2 * print_sigma * width_y, 2 * print_sigma * width_y * PIX2MM,
    2 * print_sigma * width_y * PIX2MM * MM2ARCMIN,
    print_sigma, 2 * print_sigma * width_x, 2 * print_sigma * width_x * PIX2MM,
    2 * print_sigma * width_x * PIX2MM * MM2ARCMIN,))

    plt.figure()
    gaus_arr = gaussian_baseline(*params)(*np.indices(data.shape))
    print(gaus_arr.shape, data.shape)

    from matplotlib.colors import LogNorm
    # plt.imshow(gaus_arr, cmap=plt.cm.gray, norm=LogNorm(vmin=0.01, vmax=np.max(gaus_arr)))
    # plt.imshow(gaus_arr, cmap=plt.cm.gray)
    # plt.imshow(data-gaus_arr, cmap=plt.cm.jet, vmin=-5, vmax=5)
    plt.imshow(data - gaus_arr, cmap=plt.cm.jet)
    plt.colorbar()

    chi2LC, dofLC, redchi2LC = chisq(np.ravel(data),
                                     np.ravel(gaus_arr),
                                     1,
                                     7
                                     )
    lnL_LC = -0.5 * chi2LC
    print("fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC, dofLC, redchi2LC))
    print("Log likelihood lnL={0}".format(lnL_LC))

    plt.text(0.6, 0.05,
             "baseline %.1f \n chisq %.1f \n dof %d \n red-chisq %.1f" % (baseline, chi2LC, dofLC, redchi2LC),
             fontsize=12, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes, color='k')

    plt.tight_layout()
    plt.savefig("Residual_gaussian_fit_with_baseline.pdf")
    print("baseline data {}, model {}".format(np.mean(data[80:, 80:]), baseline))
    data = gaus_arr

    print(params)
    center_crop = data[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    print("Frac contained in central image pixel from model")
    print(center_crop.sum() / data.sum())
    center_crop = data[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    print("Frac contained in central trigger pixel from model")
    print(center_crop.sum() / data.sum())

    clean_gaussian = gaus_arr - baseline
    center_crop = clean_gaussian[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    print("Frac contained in central image pixel from only the gaussian component in model")
    print(center_crop.sum() / clean_gaussian.sum())
    center_crop = clean_gaussian[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    print("Frac contained in central trigger pixel from only the gaussian component in model")
    print(center_crop.sum() / clean_gaussian.sum())

    print("Moments after subtracting basline")
    print(fitgaussian_baseline(data - baseline))
    print(moments_baseline(data - baseline))

    return fit


# fit_gaussian2d(H)
# data_fitted = fit_gaussian2d(im_best_crop, outfile="opticalPSF_2d_log_fit.pdf", df=df_best, log=True) #, amp=200, xc=0, yc=0, A=df_best.A_IMAGE[0], B=df_best.B_IMAGE[0],
# theta=df_best.THETA_IMAGE[0])



def fit_gaussian2d_baseline3(data, outfile=None, df=None, log=False,
                             show_crop=0,PIX2MM=PIX2MM, constant_baseline=0,
                             legend=False, draw_pixel=True):  # , amp=1, xc=0,yc=0,A=1,B=1,theta=0, offset=0):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # plt.matshow(data, cmap=plt.cm.gray)
    # plt.pcolor(data, cmap=plt.cm.gray)
    if log:
        from matplotlib.colors import LogNorm
        plt.imshow(data, cmap=plt.cm.gray, norm=LogNorm(vmin=0.01, vmax=np.max(data)))
    else:
        if show_crop:
            # plt.imshow(data[int(data.shape[1]/2-show_crop/2):int(data.shape[1]/2+show_crop/2),int(data.shape[0]/2-show_crop/2):int(data.shape[0]/2+show_crop/2)], cmap=plt.cm.gray)
            """
                    plt.imshow(data[int(data.shape[1]/2-show_crop/2):int(data.shape[1]/2+show_crop/2),int(data.shape[0]/2-show_crop/2):int(data.shape[0]/2+show_crop/2)], cmap=plt.cm.gray,
                    #plt.imshow(data, cmap=plt.cm.gray, 
                        extent=[int(data.shape[1]/2-show_crop/2),int(data.shape[1]/2+show_crop/2),
                                int(data.shape[0]/2-show_crop/2),int(data.shape[0]/2+show_crop/2)]
                              )
                    """
            plt.imshow(data[int(data.shape[0] / 2 - show_crop / 2):int(data.shape[0] / 2 + show_crop / 2),
                       int(data.shape[1] / 2 - show_crop / 2):int(data.shape[1] / 2 + show_crop / 2)],
                       cmap=plt.cm.gray,
                       # plt.imshow(data, cmap=plt.cm.gray,
                       extent=[int(data.shape[0] / 2 - show_crop / 2), int(data.shape[0] / 2 + show_crop / 2),
                               int(data.shape[1] / 2 + show_crop / 2), int(data.shape[1] / 2 - show_crop / 2)]
                       )
            #print(int(data.shape[0] / 2 - show_crop / 2), int(data.shape[0] / 2 + show_crop / 2),
            #      int(data.shape[1] / 2 - show_crop / 2), int(data.shape[1] / 2 + show_crop / 2))
        else:
            plt.imshow(data, cmap=plt.cm.gray)

    plt.colorbar()

    if constant_baseline > 0 :
        data = data - constant_baseline
        params = fitgaussian_0baseline(data)
        fit = gaussian_0baseline(*params)
        (height, x, y, width_x, width_y, theta) = params
        baseline = constant_baseline
    elif constant_baseline == 0:
        params = fitgaussian_baseline(data)
        fit = gaussian_baseline(*params)
        (height, x, y, width_x, width_y, theta, baseline) = params

    # plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper, levels=[68, 90, 95])
    ax = plt.gca()
    # print(height, x, y, width_x, width_y, np.rad2deg(theta))
    semi_maj = max(width_x, width_y)
    semi_min = min(width_x, width_y)
    elongation = semi_maj/semi_min
    ellipticity = 1 - (semi_min/semi_maj)
    eccentricity = np.sqrt(1 - ((semi_min**2)/(semi_maj**2)))
    print(height, x, y, width_x, width_y, theta, baseline)

    center_crop = data[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    #print("Frac contained in central image pixel from data")
    #print(center_crop.sum() / data.sum())
    center_crop = data[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    #print("Frac contained in central trigger pixel from data")
    #print(center_crop.sum() / data.sum())

    if legend:
        plt.text(0.95, 0.05, """
                x : %.1f
                y : %.1f
                $\sigma_x$ : %.1f pix \n = %.1f mm = %.1f \'
                $\sigma_y$ : %.1f pix \n = %.1f mm = %.1f \'""" % (
            y, x, width_y, width_y * PIX2MM, width_y * PIX2MM * MM2ARCMIN,
            width_x, width_x * PIX2MM, width_x * PIX2MM * MM2ARCMIN,),
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='y')
    # plt.plot([x,x], [y,y+width_y], ls='-', color='c')
    # plt.plot([x,x+width_x], [y,y], ls='-', color='c')
    # e = Ellipse(xy=np.array([x,y]), width=width_x*2,
    #            height=width_y*2, angle=-theta, linewidth=1, fill=False, alpha=0.9)
    e = Ellipse(xy=np.array([y, x]), width=width_y * 2,
                height=width_x * 2, angle=theta, linewidth=1, fill=False, alpha=0.9)

    # ax.add_artist(e)
    # e.set_color('c')

    sigma80perc = 1.794
    # e2 = Ellipse(xy=np.array([x,y]), width=width_x*4,
    #            height=width_y*4, angle=-theta, linewidth=1, fill=False, alpha=0.9)
    # below is 2 sigma
    # e2 = Ellipse(xy=np.array([y,x]), width=width_y*4,
    #            height=width_x*4, angle=theta, linewidth=1, fill=False, alpha=0.9)
    e2 = Ellipse(xy=np.array([y, x]), width=width_y * 2 * sigma80perc,
                 height=width_x * 2 * sigma80perc, angle=theta, linewidth=1, fill=False, alpha=0.9)

    ax.add_artist(e2)
    e2.set_color('c')

    sigma = 1.5
    sigma = 1.5
    ex = Ellipse(xy=np.array([y, x]), width=width_y * 2 * sigma,
                 height=width_x * 2 * sigma, angle=theta, linewidth=1, fill=False, alpha=0.9)

    # ax.add_artist(ex)
    # ex.set_color('c')

    if df is not None:
        kr = df['KRON_RADIUS']
        e3 = Ellipse(xy=np.array([y, x]), width=df['A_IMAGE'] * kr,
                     height=df['B_IMAGE'] * kr, angle=df['THETA_IMAGE'], linewidth=1, fill=False, alpha=0.9)
        print("Sextractor diameter A_mm {} B_mm {}".format(df['A_IMAGE'] * kr * PIX2MM,
                                                           df['B_IMAGE'] * kr * PIX2MM))
        # e3 = Ellipse(xy=np.array([y,x]), width=width_y*2,
        #        height=width_x*2, angle=theta, linewidth=1, fill=False, alpha=0.9)

        ax.add_artist(e3)
        e3.set_color('m')

    if draw_pixel:
        # Create a Rectangle patch
        # rect = Rectangle((70,70),25,25,#linewidth=1.5,
        # 3.7'
        # rect = Rectangle((5,65),25,25,#linewidth=1.5,
        #                 edgecolor='y',facecolor='none')
        # 4'
        # rect = Rectangle((4,62),6.53/PIX2MM,6.53/PIX2MM,#linewidth=1.5,
        pix_length = 6.53/PIX2MM
        rect = Rectangle((y - pix_length / 2, x - pix_length / 2), pix_length, pix_length,  # linewidth=1.5,
                         edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        pix_length = 6.53/PIX2MM * 2
        rect = Rectangle((y - pix_length / 2, x - pix_length / 2), pix_length, pix_length,  # linewidth=1.5,
                         edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        # plt.text(0.24, 0.02, "3.7\'",
        plt.text(0.54, 0.15, "8\'",
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='w')

    else:
        plt.plot([10, 10 + 3 / (PIX2MM * MM2ARCMIN)], [90, 90], 'y-')
        # plt.plot([10, 10+3/PIX2MM * (1./1.5)], [90,90], 'y-')
        plt.text(0.23, 0.02, "3\'",
                 fontsize=16, horizontalalignment='right',
                 verticalalignment='bottom', transform=ax.transAxes, color='y')

    plt.axis('off')
    plt.tight_layout()
    if outfile is not None:
        if outfile.split(".")[-1] == "png":
            plt.savefig(outfile, dpi=300, bbox_inches='tight',
                        pad_inches=0)
        else:
            plt.savefig(outfile, bbox_inches='tight',
                        pad_inches=0)


    plt.figure()
    if constant_baseline == 0:
        gaus_arr = gaussian_baseline(*params)(*np.indices(data.shape))
    elif constant_baseline > 0:
        gaus_arr = gaussian_0baseline(*params)(*np.indices(data.shape))

    #print(gaus_arr.shape, data.shape)

    from matplotlib.colors import LogNorm
    # plt.imshow(gaus_arr, cmap=plt.cm.gray, norm=LogNorm(vmin=0.01, vmax=np.max(gaus_arr)))
    # plt.imshow(gaus_arr, cmap=plt.cm.gray)
    # plt.imshow(data-gaus_arr, cmap=plt.cm.jet, vmin=-5, vmax=5)
    plt.imshow(data - gaus_arr, cmap=plt.cm.jet)
    plt.colorbar()

    chi2LC, dofLC, redchi2LC = chisq(np.ravel(data),
                                     np.ravel(gaus_arr),
                                     1,
                                     7
                                     )
    lnL_LC = -0.5 * chi2LC
    print("fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC, dofLC, redchi2LC))
    #print("Log likelihood lnL={0}".format(lnL_LC))

    plt.text(0.6, 0.05,
             "baseline %.1f \n chisq %.1f \n dof %d \n red-chisq %.1f" % (baseline, chi2LC, dofLC, redchi2LC),
             fontsize=12, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes, color='k')

    plt.tight_layout()
    if constant_baseline == 0:
        plt.savefig(outfile[:-4]+"_residual_gaussian_fit_with_baseline.pdf")
        print("baseline data {}, model {}".format(np.mean(data[80:, 80:]), baseline))
        clean_gaussian = gaus_arr - baseline

    elif constant_baseline > 0:
        plt.savefig(outfile[:-4]+"_residual_gaussian_fit_with_fixed_baseline.pdf")
        print("baseline data {}, input fixed value {}".format(np.mean(data[80:, 80:]), constant_baseline))
        clean_gaussian = gaus_arr

    data = gaus_arr

    #print(params)
    center_crop = data[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    print("Frac contained in central image pixel from model")
    print(center_crop.sum() / data.sum())
    center_crop = data[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    print("Frac contained in central trigger pixel from model")
    print(center_crop.sum() / data.sum())


    center_crop = clean_gaussian[int(y - 6.53/PIX2MM / 2.):int(y - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM), int(x - 6.53/PIX2MM / 2.):int(x - 6.53/PIX2MM / 2.) + int(6.53/PIX2MM)]
    print("Frac contained in central image pixel from only the gaussian component in model")
    print(center_crop.sum() / clean_gaussian.sum())
    center_crop = clean_gaussian[int(y - 6.53/PIX2MM):int(y + 6.53/PIX2MM), int(x - 6.53/PIX2MM):int(x + 6.53/PIX2MM)]
    # print(center_crop.sum(),data.sum(), center_crop.sum()/data.sum())
    print("Frac contained in central trigger pixel from only the gaussian component in model")
    print(center_crop.sum() / clean_gaussian.sum())

    #print("Moments after subtracting basline")
    #print(fitgaussian_baseline(data - baseline))
    #print(moments_baseline(data - baseline))

    print(r"""
                    x : %.1f
                    y : %.1f
                    $\sigma_x$ : %.1f pix = %.1f mm = %.2f '
                    $\sigma_y$ : %.1f pix = %.1f mm = %.2f '
                    elongation: %.2f
                    ellipticity: %.2f
                    eccentricity: %.2f
                    ========
                    The optical PSF (2 x max{$\sigma_x$, $\sigma_y$}) is: 
                    %.2f '
                    ========
                    """ % (
        y, x, abs(width_y), abs(width_y) * PIX2MM, abs(width_y) * PIX2MM * MM2ARCMIN,
        abs(width_x), abs(width_x) * PIX2MM, abs(width_x) * PIX2MM * MM2ARCMIN,
        elongation, ellipticity, eccentricity,
        2.*max(abs(width_x), abs(width_y)) * PIX2MM * MM2ARCMIN,))

    return fit, 2.*max(abs(width_x), abs(width_y)) * PIX2MM * MM2ARCMIN, \
           abs(width_y), abs(width_y) * PIX2MM, abs(width_y) * PIX2MM * MM2ARCMIN, \
           abs(width_x), abs(width_x) * PIX2MM, abs(width_x) * PIX2MM * MM2ARCMIN


def get_datetime_rawname(raw_name):
    pattern = r'\b\w{1,4}-\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2}'
    match = re.search(pattern, raw_name)
    dt_match = match.group()  # raw_name[:match.start()]
    dt_match = "_".join(dt_match.split('-'))
    dt_match = "_".join(dt_match.split(':'))
    return dt_match

def get_baseline_from_region(im_best, xmin=1580, xmax=1680, ymin=980, ymax=1080, show=False):
    im_crop = im_best[ymin:ymax, xmin:xmax]
    me_ = np.mean(im_crop)
    m_ = np.median(im_crop)
    std_ = np.std(im_crop)
    q95 = np.quantile(im_crop, 0.95)
    print("In the background region, mean = {}, median = {}, std = {}, 95% = {}".format(me_, m_, std_, q95))
    if show:
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.imshow(im_best, cmap=plt.cm.gray)
        #rect = Rectangle((ymin, xmin), ymax-ymin, xmax-xmin,  # linewidth=1.5,
        rect=Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,  # linewidth=1.5,
                                        edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        plt.show()
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.imshow(im_crop, cmap=plt.cm.gray)
        plt.show()


    return m_, std_, q95

def main():
    parser = argparse.ArgumentParser(description='Compute optical PSF')
    #parser.add_argument('rawfile', default="/home/ctauser/Pictures/Aravis/The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-2020-12-05-02:15:31.raw", type=str)
    parser.add_argument('rawfile', default="/home/ctauser/Pictures/Aravis/The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-2022-11-11-01:54:52.raw", type=str)
    parser.add_argument('--catalog', default="data/res_focal_plane_2022_11_11_01_54_52_ring_search_vvv_P1.csv", type=str)
    parser.add_argument("--psf_search_halfwidth", type=float, default=18 )
    parser.add_argument("--PIX2MM", type=float, default=0.482 )
    parser.add_argument("-o", "--outfile", dest="outfile", default=None, type=str)

    parser.add_argument("--box_bkg", action='store_true')
    parser.add_argument("--show_bkg", action='store_true')

    parser.add_argument('--bkg_x1', default=1600, type=int, )
    parser.add_argument('--bkg_x2', default=1680, type=int, )
    parser.add_argument('--bkg_y1', default=980, type=int, )
    parser.add_argument('--bkg_y2', default=1050, type=int, )

    args = parser.parse_args()

    # IO
    #fim_best = "/home/ctauser/Pictures/Aravis/The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-2020-12-05-02:15:31.raw"
    fim_best = args.rawfile
    im_best = read_raw(fim_best)

    f_best = args.catalog
    df_best = pd.read_csv(f_best)
    df_best = df_best[['Panel_ID_guess', '#', 'X_IMAGE', 'Y_IMAGE']]
    df_best["PSFarcmin"] = 0
    df_best["sigmaXpix"] = 0
    df_best["sigmaYpix"] = 0
    df_best["sigmaXmm"] = 0
    df_best["sigmaYmm"] = 0
    df_best["sigmaXarcmin"] = 0
    df_best["sigmaYarcmin"] = 0

    # if a background region is given:
    if args.box_bkg:
        print("Use box region to get background: ")
        m_, std_, q95 = get_baseline_from_region(im_best, xmin=args.bkg_x1, xmax=args.bkg_x2,
                                                 ymin=args.bkg_y1, ymax=args.bkg_y2, show=args.show_bkg)


    # cropping
    halfwidth = args.psf_search_halfwidth

    PIX2MM = args.PIX2MM
    print("The plate scale of {} mm/pixel is used".format(PIX2MM))


    """
    x_center = 1898
    y_center = 1104
    xmin = int(x_center - halfwidth)
    xmax = int(x_center + halfwidth)
    ymax = int(y_center + halfwidth)
    ymin = int(y_center - halfwidth)
    """

    for i, row in df_best.iterrows():
        ymax = int(row['Y_IMAGE'] + halfwidth)
        xmin = int(row['X_IMAGE'] - halfwidth)
        xmax = int(row['X_IMAGE'] + halfwidth)
        ymin = int(row['Y_IMAGE'] - halfwidth)

        im_best_crop = im_best[ymin:ymax, xmin:xmax]

        # computing
        """
        data_fitted = fit_gaussian2d_baseline(im_best_crop, outfile="opticalPSF_2d_log_noaxes_nolegend_80perc_baseline.pdf",
                                              # draw_pixel=False,
                                              # legend=True,
                                              # df=df_best,
                                              log=False)  # , amp=200, xc=0, yc=0, A=df_best.A_IMAGE[0], B=df_best.B_IMAGE[0],
        """

        dt_match = get_datetime_rawname(args.rawfile)

        if args.box_bkg:
            data_fitted, PSF, sigma_x,sigma_xmm,sigma_xarcmin, \
                sigma_y,sigma_ymm,sigma_yarcmin = fit_gaussian2d_baseline3(im_best_crop,
                                                   outfile="data/opticalPSF_{}_{}.pdf".format(dt_match, row['Panel_ID_guess']),
                                                   PIX2MM=PIX2MM,
                                                   constant_baseline = m_,
                                                   # draw_pixel=False,
                                                   # legend=True,
                                                   # df=df_best,
                                                   log=False)  # , amp=200, xc=0, yc=0, A=df_best.A_IMAGE[0], B=df_best.B_IMAGE[0],
        else:
            data_fitted, PSF, sigma_x,sigma_xmm,sigma_xarcmin, \
                sigma_y,sigma_ymm,sigma_yarcmin  = fit_gaussian2d_baseline3(im_best_crop, outfile="data/opticalPSF_{}_{}.pdf".format(dt_match, row['Panel_ID_guess']),PIX2MM=PIX2MM,
                                           # draw_pixel=False,
                                           # legend=True,
                                           #df=df_best,
                                           log=False)  # , amp=200, xc=0, yc=0, A=df_best.A_IMAGE[0], B=df_best.B_IMAGE[0],
        df_best.loc[i, "PSFarcmin"] = PSF
        df_best.loc[i, "sigmaXpix"] = sigma_x
        df_best.loc[i, "sigmaYpix"] = sigma_xmm
        df_best.loc[i, "sigmaXmm"] = sigma_xarcmin
        df_best.loc[i, "sigmaYmm"] = sigma_y
        df_best.loc[i, "sigmaXarcmin"] = sigma_ymm
        df_best.loc[i, "sigmaYarcmin"] = sigma_yarcmin

    if args.outfile is not None:
        #df_best.to_csv(args.outfile + "_single_panel_PSF.csv", index=False)
        #pd.options.display.float_format = "{:.2f}".format #doesnt work
        cols = ["PSFarcmin", "sigmaXpix", "sigmaYpix", "sigmaXmm", "sigmaYmm", "sigmaXarcmin", "sigmaYarcmin"]
        df_best[cols] = df_best[cols].applymap('{:.2f}'.format)
        df_best.to_csv(args.outfile, index=False)
        print("Single panel PSF saved to file {}".format(args.outfile))

if __name__ == '__main__':
    main()
