import cv2


class Camera(object):
    def __init__(self):
        self.input = cv2.VideoCapture(0)

    def get_camera(self, mirror=False):
        img = self.input.read()[1]
        if mirror:
            img = cv2.flip(img, 1)

        return img
