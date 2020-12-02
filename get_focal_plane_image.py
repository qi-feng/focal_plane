#! /usr/bin/python
import sys

sys.path.append("/usr/alignment/opencv-3.4.3/lib/python2.7/site-packages/")
try:
    import gi
    gi.require_version('Aravis', '0.6')
    from gi.repository import Aravis
    import cv2
except:
    print("which executes \"export GI_TYPELIB_PATH=$GI_TYPELIB_PATH:/home/ctauser/skycam/aravis/src\"")
    print("and \"export LD_LIBRARY_PATH=/home/ctauser/skycam/aravis/src/.libs\"")

import ctypes
import numpy as np

from datetime import datetime
from pytz import timezone
import time
import logging
import argparse
import subprocess
import os
import re

# Let's try ignore DeprecationWarnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# input timeout in case the script is run during the day
import select


def setup_logger(logger_name, log_file, level=logging.INFO, show_log=True):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(file_handler)
    log_setup.addHandler(stream_handler)
    if show_log:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log_setup.addHandler(ch)


def get_device_ids():
    Aravis.update_device_list()
    n = Aravis.get_n_devices()
    return [Aravis.get_device_id(i) for i in range(0, n)]


def show_frame(frame):
    cv2.imshow("capture", frame)
    cv2.waitKey(0)


def save_frame(frame, path="frame.jpeg"):
    print("Saving frame to ", path)
    np.save(path, frame)


def sfn(cam, path="frame.jpeg"):
    from PIL import Image
    cam.start_acquisition()
    frame = cam.pop_frame()
    cam.stop_acquisition()
    im = Image.fromarray(frame)
    print("Saving image to ", path)
    im.save(path)


def get_frame(cam):
    cam.start_acquisition()
    frame = cam.pop_frame()
    cam.stop_acquisition()
    return frame


def convert(buf):
    if not buf:
        return None
    pixel_format = buf.get_image_pixel_format()
    bits_per_pixel = pixel_format >> 16 & 0xff
    if bits_per_pixel == 8:
        INTP = ctypes.POINTER(ctypes.c_uint8)
    else:
        INTP = ctypes.POINTER(ctypes.c_uint16)
    addr = buf.get_data()
    ptr = ctypes.cast(addr, INTP)
    im = np.ctypeslib.as_array(ptr, (buf.get_image_height(), buf.get_image_width()))
    im = im.copy()
    return im


class AravisException(Exception):
    pass


class Camera(object):
    """
    Create a Camera object.
    name is the camera ID in aravis.
    If name is None, the first found camera is used.
    If no camera is found an AravisException is raised.
    """

    def __init__(self, name=None, exposure=500000, gain=15, framerate=10, loglevel=logging.WARNING):
        self.logger = logging.getLogger(self.__class__.__name__)
        if len(logging.root.handlers) == 0:  # dirty hack
            logging.basicConfig()
        self.logger.setLevel(loglevel)
        self.name = name
        self.camera_name = 'The Imaging Source Europe GmbH-DMK 23GP031-37514083'
        if name is None:
            self.name = self.camera_name
        try:
            self.cam = Aravis.Camera.new(name)
        except TypeError:
            if name:
                raise AravisException("Error the camera %s was not found or it's being used by another process", name)
            else:
                raise AravisException("Error no camera found")
        # self.name = self.cam.get_model_name()
        self.logger.info("Camera object created for device: %s", self.name)
        self.dev = self.cam.get_device()
        self.stream = self.cam.create_stream(None, None)
        if self.stream is None:
            raise AravisException("Error creating buffer")
        self._frame = None
        self._last_payload = 0
        self.get_exposure()
        self.get_framerate()
        self.get_gain()
        if self.exposure != exposure:
            self.set_exposure(exposure)
        if self.gain != gain:
            self.set_gain(gain)
        if self.framerate != framerate:
            self.set_framerate(framerate)

    def __getattr__(self, name):
        if hasattr(self.cam, name):  # expose methods from the aravis camera object which is also relatively high level
            return getattr(self.cam, name)
        # elif hasattr(self.dev, name): #epose methods from the aravis device object, this might be confusing
        #    return getattr(self.dev, name)
        else:
            raise AttributeError(name)

    def __dir__(self):
        tmp = list(self.__dict__.keys()) + self.cam.__dir__()  # + self.dev.__dir__()
        return tmp

    def get_exposure(self):
        self.exposure = self.cam.get_exposure_time()
        return self.exposure

    def set_exposure(self, exposure):
        self.cam.set_exposure_time(exposure)
        self.exposure = self.get_exposure_time()
        self.logger.info("Setting exposure time to {} us".format(self.exposure))

    def get_framerate(self):
        self.framerate = self.cam.get_frame_rate()
        return self.framerate

    def set_framerate(self, framerate):
        self.cam.set_frame_rate(framerate)
        self.framerate = self.get_framerate()
        self.logger.info("Setting frame rate to {} Hz".format(self.framerate))

    def get_gain(self):
        self.gain = self.cam.get_gain()
        return self.gain

    def set_gain(self, gain):
        self.cam.set_gain(gain)
        self.gain = self.get_gain()
        self.logger.info("Setting gain to {}".format(self.gain))

    def get_feature_type(self, name):
        genicam = self.dev.get_genicam()
        node = genicam.get_node(name)
        if not node:
            raise AravisException("Feature {} does not seem to exist in camera".format(name))
        return node.get_node_name()

    def get_feature(self, name):
        """
        return value of a feature. independently of its type
        """
        ntype = self.get_feature_type(name)
        if ntype in ("Enumeration", "String", "StringReg"):
            return self.dev.get_string_feature_value(name)
        elif ntype == "Integer":
            return self.dev.get_integer_feature_value(name)
        elif ntype == "Float":
            return self.dev.get_float_feature_value(name)
        elif ntype == "Boolean":
            return self.dev.get_integer_feature_value(name)
        else:
            self.logger.warning("Feature type not implemented: %s", ntype)

    def set_feature(self, name, val):
        """
        set value of a feature
        """
        ntype = self.get_feature_type(name)
        if ntype in ("String", "Enumeration", "StringReg"):
            return self.dev.set_string_feature_value(name, val)
        elif ntype == "Integer":
            return self.dev.set_integer_feature_value(name, int(val))
        elif ntype == "Float":
            return self.dev.set_float_feature_value(name, float(val))
        elif ntype == "Boolean":
            return self.dev.set_integer_feature_value(name, int(val))
        else:
            self.logger.warning("Feature type not implemented: %s", ntype)

    def get_genicam(self):
        """
        return genicam xml from the camera
        """
        return self.dev.get_genicam_xml()

    def get_feature_vals(self, name):
        """
        if feature is an enumeration then return possible values
        """
        ntype = self.get_feature_type(name)
        if ntype == "Enumeration":
            return self.dev.get_available_enumeration_feature_values_as_strings(name)
        else:
            raise AravisException("{} is not an enumeration but a {}".format(name, ntype))

    def read_register(self, address):
        return self.dev.read_register(address)

    def write_register(self, address, val):
        return self.dev.write_register(address, val)

    def create_buffers(self, nb=10, payload=None):
        if not payload:
            payload = self.cam.get_payload()
        self.logger.info("Creating %s memory buffers of size %s", nb, payload)
        for _ in range(0, nb):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    def pop_frame(self, timestamp=False):
        while True:  # loop in python in order to allow interrupt, have the loop in C might hang
            if timestamp:
                ts, frame = self.try_pop_frame(timestamp)
            else:
                frame = self.try_pop_frame()

            if frame is None:
                time.sleep(0.001)
            else:
                if timestamp:
                    return ts, frame
                else:
                    return frame

    def try_pop_frame(self, timestamp=False):
        """
        return the oldest frame in the aravis buffer
        """
        buf = self.stream.try_pop_buffer()
        if buf:
            frame = self.__array_from_buffer_address(buf)
            self.stream.push_buffer(buf)
            if timestamp:
                # return buf.get_timestamp(), frame
                timestamp_az = timezone('America/Phoenix').localize(
                    datetime.fromtimestamp(buf.get_system_timestamp() * 1e-9))
                timestamp_utc = timestamp_az.astimezone(timezone('UTC'))
                # return datetime.fromtimestamp(buf.get_system_timestamp()*1e-9).strftime("%Y-%m-%d_%H:%M:%S.%f"), frame
                return timestamp_utc.strftime("%Y-%m-%d-%H:%M:%S"), frame
            else:
                return frame
        else:
            if timestamp:
                return None, None
            else:
                return None

    def try_save_frame(self, timestamp=True, work_dir='./', save_filename=None):
        if save_filename is None:
            # timestamp, frame = self.try_pop_frame(timestamp=timestamp)
            timestamp, frame = self.pop_frame(timestamp=timestamp)
            self.logger.debug(timestamp, frame)
            # example name: The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-2020-12-02-01:46:38.raw
            outfile = os.path.join(work_dir, "The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-{}.jpg".format(timestamp))
            if buffer:
                cv2.imwrite(outfile, frame)
            else:
                self.logger.error("*** Failed to save image skycam_image{}.jpeg ***".format(timestamp))
            return outfile, timestamp
        else:
            frame = self.pop_frame(timestamp=False)
            self.logger.debug(frame)
            # example name: The Imaging Source Europe GmbH-37514083-2592-1944-Mono8-2020-12-02-01:46:38.raw
            outfile = os.path.join(work_dir, save_filename)
            if buffer:
                cv2.imwrite(outfile, frame)
            else:
                self.logger.error("*** Failed to save image {} ***".format(save_filename))
            return outfile

    def __array_from_buffer_address(self, buf):
        if not buf:
            return None
        pixel_format = buf.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xff
        if bits_per_pixel == 8:
            INTP = ctypes.POINTER(ctypes.c_uint8)
        else:
            INTP = ctypes.POINTER(ctypes.c_uint16)
        addr = buf.get_data()
        ptr = ctypes.cast(addr, INTP)
        im = np.ctypeslib.as_array(ptr, (buf.get_image_height(), buf.get_image_width()))
        im = im.copy()
        return im

    def trigger(self):
        """
        trigger camera to take a picture when camera is in software trigger mode
        """
        self.execute_command("TriggerSoftware")

    def __str__(self):
        return "Camera: " + self.name

    def __repr__(self):
        return self.__str__()

    def start_acquisition(self, nb_buffers=10):
        self.logger.info("starting acquisition")
        payload = self.cam.get_payload()
        if payload != self._last_payload:
            # FIXME should clear buffers
            self.create_buffers(nb_buffers, payload)
            self._last_payload = payload
        self.cam.start_acquisition()

    def start_acquisition_trigger(self, nb_buffers=1):
        self.set_feature("AcquisitionMode", "Continuous")  # no acquisition limits
        self.set_feature("TriggerSource", "Software")  # wait for trigger t acquire image
        self.set_feature("TriggerMode", "On")  # Not documented but necessary
        self.start_acquisition(nb_buffers)

    def start_acquisition_continuous(self, nb_buffers=20):
        self.set_feature("AcquisitionMode", "Continuous")  # no acquisition limits
        self.start_acquisition(nb_buffers)

    def stop_acquisition(self):
        self.cam.stop_acquisition()


def get_image_from_cam(args):
    setup_logger("camera_log", os.path.join(args.work_dir, args.log_file), level=logging.INFO, show_log=args.verbose)
    camera_logger = logging.getLogger('camera_log')

    cam = Camera(name='The Imaging Source Europe GmbH-DMK 23GP031-37514083', loglevel=logging.WARNING,
                 exposure=args.exposure, gain=args.gain, framerate=args.framerate)

    timestamp_az = timezone('America/Phoenix').localize(datetime.now())
    timestamp_utc = timestamp_az.astimezone(timezone('UTC'))
    workdir = args.data_outdir

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    try:
        cam.start_acquisition(1)
        if args.data_outfile is None:
            figfile, timestamp = cam.try_save_frame(work_dir=workdir)
            figfile = os.path.join(workdir, figfile)
            if timestamp is None:
                camera_logger.info("Trouble saving frame, trying again...")
                cam.stop_acquisition()
                cam.pop_frame()
            else:
                camera_logger.info("Frame saved to {}".format(figfile))
        else:
            figfile = cam.try_save_frame(work_dir=workdir, save_filename=args.data_outfile)
            camera_logger.info("Frame saved to {}".format(figfile))
        cam.stop_acquisition()

    finally:
        camera_logger.info("Stopping camera acquisition")
        cam.stop_acquisition()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download an image from camera stream')

    # parser.add_argument('-i', '--input', default=None, help='Input raw image file; default is None so that camera stream is used. ')
    parser.add_argument('-w', '--work_dir', default="/home/ctauser/Pictures/Aravis", help='work dir')
    parser.add_argument('-l', '--log_file', default="camera_log.txt", help='log file name')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose flag')
    parser.add_argument('-e', '--exposure', default=500000, help="Exposure time of the camera, default is 500 ms, or 500000 us", type=float)
    parser.add_argument('-g', '--gain', default=15, help="Gain of the camera, default is 15", type=float)
    parser.add_argument('-f', '--framerate', default=10, help="Frame rate of the camera, default is 10", type=float)
    parser.add_argument('--data_outdir', default='/home/ctauser/Pictures/Aravis')
    parser.add_argument('--data_outfile', default=None, help='Image file path to create. Default is None. ')

    args = parser.parse_args()

    get_image_from_cam(args)
