import win32api
import win32gui
import win32ui

import cv2
import numpy as np
import win32con

from classes.camera import Camera


class Screen(object):
    def __init__(self):
        self.hwin = win32gui.GetDesktopWindow()

    # Screen Grab based on supplied region
    # Returns converted image

    def grab_screen(self, region=None, preprocess=False):
        if region:
            left, top, x2, y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        try:
            hwindc = win32gui.GetWindowDC(self.hwin)
            srcdc = win32ui.CreateDCFromHandle(hwindc)
            memdc = srcdc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(srcdc, width, height)
            memdc.SelectObject(bmp)
            memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

            signed_ints_array = bmp.GetBitmapBits(True)
            capture = np.fromstring(signed_ints_array, dtype='uint8')
            capture.shape = (height, width, 4)

            srcdc.DeleteDC()
            memdc.DeleteDC()
            win32gui.ReleaseDC(self.hwin, hwindc)
            win32gui.DeleteObject(bmp.GetHandle())
            if preprocess is True:
                capture = cv2.cvtColor(capture, cv2.COLOR_BGRA2GRAY)
                # capture = cv2.threshold(capture, 127, 255, 0)[1]
                # capture = cv2.applyColorMap(capture, cv2.COLORMAP_RAINBOW)
                # capture = cv2.Canny(capture, threshold1=100, threshold2=300)
                contours = cv2.findContours(capture, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
                contour_img = cv2.drawContours(capture, contours, 2, (255, 0, 0), 1)
                # capture = cv2.blur(capture, (7, 7))
                capture = capture + contour_img

            else:
                capture = cv2.cvtColor(capture, cv2.COLOR_BGRA2RGB)

            return capture
        except Exception as e:
            print('Error: %s' % e)
