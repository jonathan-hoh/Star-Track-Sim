# calibrate.py - bast
#
# Python module for preparing calibration for running basic astronimical star tracking (bast)


class Calibration:
    def __init__(self, dark_frame=None, distortion_map=None):
        self.dark_frame = dark_frame
        self.distorion_map = distortion_map
