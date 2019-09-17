pSCT optical alignment using optical images of a star at the focal plane

# Intro
Two python scripts are in this repo, they are quickly put together and not smart, user be aware. 
To identify a panel motion in the focal plane image, first run focal_plane.py to create catalog of centroids, and then identify by eye the number (as shown in the pdf for centroids that are not common in both imamges) of the first (green) centroid and the second (yellow) centroid, finally run find_motion_focal_plane.pyto produce a clean output of the information of these two centroid. 

# Prerequisite 

sextractor, sewpy, numpy, pandas, matplotlib 
Ideally, opencv2.4 (cv2)

# Usage
## focal_plane.py

** This code processes the raw images using Sextractor, create catalog of centroids, and isolate unique centroids in image1 and image2 if the distance between them is larger than min_dist, make "diff_cat" plot with unique sources in the 1st image labelled in green, and those in the 2nd image labelled in yellow. Common images are shown as red (or not shown in the "diff_cat" plots). ** 

The easiest example to run focal_plane.py is 
```
python focal_plane.py rawfile1 rawfile2 --DETECT_MINAREA DETECT_MINAREA --THRESH THRESH --min_dist MIN_DIST
```
e.g., 
```
python focal_plane.py The\ Imaging\ Source\ Europe\ GmbH-37514083-2592-1944-Mono8-2019-09-16-01\:25\:51.raw The\ Imaging\ Source\ Europe\ GmbH-37514083-2592-1944-Mono8-2019-09-16-01\:26\:42.raw --min_dist 20 --THRESH 2 --DETECT_MINAREA 80
```

The most important parameters are listed: 

- min_dist: the minimum distance between two centroid to identify motion, this will depend on how large a motion the panel was moved. If panel motion is large, can use something like 10 to 20 (pixels) to reduce false alarms (which will happen anyways); if a panel motion is small, use small values as low as 2 pixels. 

- THRESH: Sextractor image centroid analysis threshold, if you see your suspect centroid but it's not identified, lower this number. Depending on the light condition and exposure of the image, something between 2 to 8 have been useful, in extreme cases may need values out of this range. 

- DETECT_MINAREA: Sextractor minimum number of pixels above threshold triggering detection. 

## find_motion_focal_plane.py

** This code is run after focal_plane.py, if you believe you found two centroids that correspond to the same panel, use the green number in diff_cat1 image for this centroid and the yellow number in diff_cat2 image, and run this with -1 and -2 options. 

The easiest way to run find_motion_focal_plane.py is 

```
python find_motion_focal_plane.py rawfile1 rawfile2 -1 centroid_number1 -2 centroid_number2
```

e.g., 
```
python find_motion_focal_plane.py The\ Imaging\ Source\ Europe\ GmbH-37514083-2592-1944-Mono8-2019-09-16-01\:25\:51.raw The\ Imaging\ Source\ Europe\ GmbH-37514083-2592-1944-Mono8-2019-09-16-01\:26\:42.raw -1 11 -2 20
```


Full use of focal_plane.py can be found using: 

```
python focal_plane.py -h
usage: focal_plane.py [-h] [--DETECT_MINAREA DETECT_MINAREA] [--THRESH THRESH]
                      [--kernel_w KERNEL_W] [--min_dist MIN_DIST]
                      [--save_filename_prefix1 SAVE_FILENAME_PREFIX1]
                      [--save_filename_prefix2 SAVE_FILENAME_PREFIX2]
                      [--savefits_name1 SAVEFITS_NAME1]
                      [--saveplot_name1 SAVEPLOT_NAME1]
                      [--savefits_name2 SAVEFITS_NAME2]
                      [--saveplot_name2 SAVEPLOT_NAME2]
                      [--savecatalog_name1 SAVECATALOG_NAME1]
                      [--savecatalog_name2 SAVECATALOG_NAME2]
                      [--diffcatalog_name1 DIFFCATALOG_NAME1]
                      [--diffcatalog_name2 DIFFCATALOG_NAME2]
                      [--diffplot_name1 DIFFPLOT_NAME1]
                      [--diffplot_name2 DIFFPLOT_NAME2] [--cropx1 CROPX1]
                      [--cropx2 CROPX2] [--cropy1 CROPY1] [--cropy2 CROPY2]
                      [-o MOTION_OUTFILE_PREFIX] [--nozoom] [-v]
                      rawfile1 rawfile2

Compare two raw images of stars at the focal plane

positional arguments:
  rawfile1
  rawfile2

optional arguments:
  -h, --help            show this help message and exit
  --DETECT_MINAREA DETECT_MINAREA
                        +++ Important parameter +++: Config param for
                        sextractor, our default is 30.
  --THRESH THRESH       +++ Important parameter +++: Config param for
                        sextractor, our default is 6.
  --kernel_w KERNEL_W   If you have cv2, this is the median blurring kernel
                        widthour default is 3 (for a 3x3 kernel).
  --min_dist MIN_DIST   +++ Important parameter +++: Minimum distance we use
                        to conclude a centroid is common in both imagesour
                        default is 20 pixels.
  --save_filename_prefix1 SAVE_FILENAME_PREFIX1
                        File name prefix of the output files, this will
                        automatically populate savefits_name1, saveplot_name1,
                        savecatalog_name1, and diffcatalog_name1default is
                        None.
  --save_filename_prefix2 SAVE_FILENAME_PREFIX2
                        File name prefix of the output files, this will
                        automatically populate savefits_name2, saveplot_name2,
                        savecatalog_name2, and diffcatalog_name2default is
                        None.
  --savefits_name1 SAVEFITS_NAME1
                        File name of the fits for the first image if you want
                        to choose, default has the same name as raw.
  --saveplot_name1 SAVEPLOT_NAME1
                        File name of the image (jpeg or pdf etc) for the first
                        image if you want to choose, default is not to save
                        it.
  --savefits_name2 SAVEFITS_NAME2
                        File name of the fits for the second image if you want
                        to choose, default has the same name as raw.
  --saveplot_name2 SAVEPLOT_NAME2
                        File name of the image (jpeg or pdf etc) for the
                        second image if you want to choose, default is not to
                        save it.
  --savecatalog_name1 SAVECATALOG_NAME1
                        File name of the ascii catalog derived from the first
                        image, default is not to save it.
  --savecatalog_name2 SAVECATALOG_NAME2
                        File name of the ascii catalog derived from the second
                        image, default is not to save it.
  --diffcatalog_name1 DIFFCATALOG_NAME1
                        File name of the ascii catalog for sources only in the
                        first image, default is diff_cat1.txt.
  --diffcatalog_name2 DIFFCATALOG_NAME2
                        File name of the ascii catalog for sources only in the
                        second image, default is diff_cat2.txt.
  --diffplot_name1 DIFFPLOT_NAME1
                        File name of the image catalog for sources only in the
                        first image, default is diff_cat1.pdf.
  --diffplot_name2 DIFFPLOT_NAME2
                        File name of the image catalog for sources only in the
                        second image, default is diff_cat2.pdf.
  --cropx1 CROPX1       zooming into xlim that you want to plot, use None for
                        no zoom, default is (1650, 2100).
  --cropx2 CROPX2       zooming into xlim that you want to plot, use None for
                        no zoom, default is (1650, 2100).
  --cropy1 CROPY1       zooming into ylim that you want to plot, use None for
                        no zoom, default is (1250, 800).
  --cropy2 CROPY2       zooming into ylim that you want to plot, use None for
                        no zoom, default is (1250, 800).
  -o MOTION_OUTFILE_PREFIX, --motion_outfile_prefix MOTION_OUTFILE_PREFIX
                        File name prefix of the output catalog to calculate
                        motion.
  --nozoom              Do not zoom/crop the image.
  -v, --verbose
```

Full use of find_motion_focal_plane.py and be found using: 

```
python find_motion_focal_plane.py -h
usage: find_motion_focal_plane.py [-h] [--diffcatalog_name1 DIFFCATALOG_NAME1]
                                  [--diffcatalog_name2 DIFFCATALOG_NAME2]
                                  [-1 IND1] [-2 IND2]
                                  [-o MOTION_OUTFILE_PREFIX]
                                  [--saveplot_name1 SAVEPLOT_NAME1]
                                  [--saveplot_name2 SAVEPLOT_NAME2]
                                  [--cropx1 CROPX1] [--cropx2 CROPX2]
                                  [--cropy1 CROPY1] [--cropy2 CROPY2]
                                  rawfile1 rawfile2

Compare two raw images of stars at the focal plane

positional arguments:
  rawfile1
  rawfile2

optional arguments:
  -h, --help            show this help message and exit
  --diffcatalog_name1 DIFFCATALOG_NAME1
                        File name of the ascii catalog for sources only in the
                        first image. If not provided, will use default that
                        focal_plane.py usesi.e.,
                        res_focal_plane_YYYY_MM_DD_HH_MM_SS_diff_cat1.txt
  --diffcatalog_name2 DIFFCATALOG_NAME2
                        File name of the ascii catalog for sources only in the
                        second image. If not provided, will use default that
                        focal_plane.py usesi.e.,
                        res_focal_plane_YYYY_MM_DD_HH_MM_SS_diff_cat2.txt
  -1 IND1, --ind1 IND1  Index of the centroid in the first catalog
  -2 IND2, --ind2 IND2  Index of the centroid in the second catalog
  -o MOTION_OUTFILE_PREFIX, --motion_outfile_prefix MOTION_OUTFILE_PREFIX
                        File name prefix of the output catalog to calculate
                        motion.
  --saveplot_name1 SAVEPLOT_NAME1
                        File name of the image (jpeg or pdf etc) for the first
                        image.
  --saveplot_name2 SAVEPLOT_NAME2
                        File name of the image (jpeg or pdf etc) for the
                        second image if you want to choose.
  --cropx1 CROPX1       zooming into xlim that you want to plot, use None for
                        no zoom, default is (1650, 2100).
  --cropx2 CROPX2       zooming into xlim that you want to plot, use None for
                        no zoom, default is (1650, 2100).
  --cropy1 CROPY1       zooming into ylim that you want to plot, use None for
                        no zoom, default is (1250, 800).
  --cropy2 CROPY2       zooming into ylim that you want to plot, use None for
                        no zoom, default is (1250, 800).

```

## calc_motion.py

** Utilities to calculate rx ry resp matrix given (dx1, dy1, rx, dx2, dy2, ry),
where dx1 and dy1 is the motion of centroid when panel rx is introduced, and
dx2 and dy2 is the motion of centroid when panel ry is introduced. Or
calculate motion needed to go to center and pattern position for a given
panel, need to provide current coordinates in camera x and y. **

- To calculate rx ry response matrix, run it like: 
```
python calc_motion.py 1424 --dx1 -12.4 --dy1 16.5 --rx 0.32 --dx2 19.4 --dy2 15.1 --ry 0.32
```
You'll be given a chance to save it to the default yaml file rx_ry_matrix.yaml. 

- For those panels that already have matrices in the file "rx_ry_matrix.yaml", to calculate the motion needed to move the image to the center of the FoV, run it like: 
```
python calc_motion.py 1328 -c -x 1444 -y 877
```
where x and y are the current centroid coordinates. 

- Similarly, if a panel has resp matrix in the file, to calculate the rx ry motion needed to move the image to the "pattern position" where all panels are spread out (default file is pattern_position.txt), run like: 
```
python calc_motion.py 1328 -p -x 1444 -y 877
```


