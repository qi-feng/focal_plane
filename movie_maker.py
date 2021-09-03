import argparse
import os
import subprocess

from focal_plane import read_raw, get_datetime_rawname
import matplotlib.pyplot as plt

try:
    import cv2
    has_cv2 = True
except:
    print("Can't import cv2!!")
    has_cv2 = False

def images2gif(im_list, output_gif):
    str_im_list = ' '.join(im_list)
    str1 = 'convert -delay 50 -loop 1 -quality 50 -density 144 ' + str_im_list +' '+output_gif
    print(str1)
    subprocess.call(str1, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Makes a movie out of the .RAW files passed as arguments."
                                                 "Use a list of RAW files with the space escaped by \'\\\' in the name."
                                                 "You can read a text file of these names with `awk -F\" '{printf(\"\%s "
                                                 "\",$1)}' test.txt`, then copy/paste that chunk of text into this call.")

    parser.add_argument("RAW_files", type=str, default=None, nargs="*")
    parser.add_argument('--output', type=str, default='movie_maker_animation.gif', help="Name of output file to make animation.")

    parser.add_argument('--kernel_w', type=int, default=3,
                        help="If you have cv2, this is the median blurring kernel width"
                             "our default is 3 (for a 3x3 kernel).")
    parser.add_argument('--datadir', default="data",
                        help="Folder to save all output files. Default is ./data (ignored by git)")
    parser.add_argument('--gifname', default=None,  # default="compare.gif",
                        help="File name to save gif animation. ")

    parser.add_argument('--cropx1', default=1050,  # default=1170,
                        # default=(1350, 1800),
                        type=int,
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")
    parser.add_argument('--cropx2', default=2592,  # default=1970,
                        # default=(1350, 1800),
                        type=int,
                        help="zooming into xlim that you want to plot, use None for no zoom, default is (1650, 2100).")

    parser.add_argument('--cropy1',  # default=1410,
                        default=1850, type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")
    parser.add_argument('--cropy2',  # default=610,
                        default=250,
                        type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom, default is (1250, 800).")
    parser.add_argument('--nozoom', action='store_true', help="Do not zoom/crop the image. ")

    args = parser.parse_args()

    if args.nozoom:
        cropxs = None
        cropys = None
    else:
        cropxs = (args.cropx1, args.cropx2)
        cropys = (args.cropy1, args.cropy2)

    png_list = []
    for single_file in args.RAW_files:
        im_raw = read_raw(single_file)
        if not args.nozoom:
            im_raw = im_raw[cropys[1]:cropys[0], cropxs[0]:cropxs[1]]
        print("Read file {}".format(single_file))
        if has_cv2:
            median = cv2.medianBlur(im_raw, args.kernel_w)
        else:
            print("+++ System doesn't have opencv installed, using noisy raw image without median blurring +++")
            median = im_raw

        dt_match = get_datetime_rawname(single_file)
        single_outfile = os.path.join(args.datadir, "res_focal_plane_" + dt_match + ".png")
        print(single_outfile)

        if has_cv2:
            cv2.imwrite(single_outfile, median)
        else:
            if cropxs is not None:
                plt.xlim(cropxs)
            if cropys is not None:
                plt.ylim(cropys)
            plt.imshow(median, cmap='gray')
            plt.savefig(single_outfile)

        png_list.append(single_outfile)

    print("Output: " + args.datadir + args.output)
    output_gif = os.path.join(args.datadir, args.output)
    images2gif(png_list, output_gif)

if __name__ == "__main__":
    main()
