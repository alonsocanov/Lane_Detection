import cv2
import numpy as np


class Camera:
    def __init__(self, file_path) -> None:
        self.__path = file_path

    def gray(self, img: np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussianBlur(self, img, kernel: tuple = (7, 7), skip: int = 0):
        return cv2.GaussianBlur(img, kernel, skip)

    def cannyEdges(self, img):
        return cv2.Canny(img, 100, 200)

    def hsv2CvHsv(self, hsv: np.ndarray) -> np.ndarray:
        '''
        Color normalization of HSV to OpenCV HSV
        For HSV, Hue range is [0,179], Saturation range is [0,255]
        and Value range is [0,255]. Different software use different scales.
        So if you are comparing in OpenCV values with them, you need to normalize these ranges.
        '''
        hsv_cv = np.array([179, 255, 255])
        hsv_orig = np.array([360, 100, 100])
        cv_hsv = np.divide((hsv * hsv_cv), hsv_orig)
        return cv_hsv

    def videoCapure(self):
        return cv2.VideoCapture(self.__path)

    def check(self, c: str = 'q') -> bool:
        '''check if key q was pressed'''
        if cv2.waitKey(1) & 0xFF == ord(c):
            return True
        return False

    def resize(self, img: np.array, factor: float = None) -> np.array:
        h, w = img.shape[:2]
        if not factor:
            factor = 300 / h
        return cv2.resize(img, (int(factor * w), int(factor * h)))

    def prespectiveTransform(self, source_pts, destination_pts):
        return cv2.getPerspectiveTransform(source_pts, destination_pts)
