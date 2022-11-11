import argparse
import matplotlib.pyplot as plt
import numpy as np
try:
    import cv2
    has_cv2 = True
except:
    print("Can't import cv2!!")
    has_cv2 = False

def read_raw(f='./GAS_image.raw', cols=2592, rows=1944, outfile=None, show=False,
             cropxs=(0, 2592), cropys=(1944, 0),
             ):
    fd = open(f, 'rb')
    f = np.fromfile(fd, dtype=np.uint8, count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    #im = im[cropys[1]:cropys[0], cropxs[0]:cropxs[1]]
    fd.close()
    if outfile is not None:
        if has_cv2 and outfile.split('.')[-1] != 'pdf':
            cv2.imwrite(outfile, im[cropys[1]:cropys[0], cropxs[0]:cropxs[1]])
        else:
            plt.imshow(im, cmap='gray')
            plt.xlim(cropxs[0],cropxs[1])
            plt.ylim(cropys[1],cropys[0])
            plt.savefig(outfile)
    if show:
        plt.imshow(im, cmap='gray')
        plt.xlim(cropxs[0], cropxs[1])
        plt.ylim(cropys[1], cropys[0])
    return im

def main():
    parser = argparse.ArgumentParser(description='convert raw file to image file')

    parser.add_argument('rawfile', type=str)
    parser.add_argument("-o", "--outfile", dest="outfile", default=None, type=str)
    parser.add_argument("--show", action='store_true')
    parser.add_argument('--cropx1', default=0, type=int,  # default=1170, 580
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom.")
    parser.add_argument('--cropx2', default=2592, type=int,  # default=1970,
                        # default=(1350, 1800),
                        help="zooming into xlim that you want to plot, use None for no zoom.")

    parser.add_argument('--cropy1',  # default=1410,
                        default=1944, type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom.")
    parser.add_argument('--cropy2',  # default=610, 150
                        default=0, type=int,
                        help="zooming into ylim that you want to plot, use None for no zoom.")

    args = parser.parse_args()

    _ = read_raw(f=args.rawfile, cols=2592, rows=1944, outfile=args.outfile, show=args.show,
             cropxs=(args.cropx1, args.cropx2), cropys=(args.cropy1, args.cropy2),
             )

if __name__ == "__main__":
    main()

