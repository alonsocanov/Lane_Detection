import cv2
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt


# color normalization of HSV to OpenCV HSV
def hsv2cvhsv(hsv: np.ndarray) -> np.ndarray:
    # For HSV, Hue range is [0,179], Saturation range is [0,255]
    # and Value range is [0,255]. Different software use different scales.
    # So if you are comparing in OpenCV values with them, you need to normalize these ranges.
    hsv_cv = np.array([179, 255, 255])
    hsv_orig = np.array([360, 100, 100])
    cv_hsv = np.divide((hsv * hsv_cv), hsv_orig)
    return cv_hsv


# check available files in path
def availableFiles(path: str) -> str:
    full_path = glob.glob(path, recursive=True)
    if full_path:
        file = full_path[0]
    else:
        file = None
    return file


# check if key q was pressed
def check(c: str = 'q') -> bool:
    if cv2.waitKey(1) & 0xFF == ord(c):
        return True
    return False


def resize(img: np.array, factor: float = None) -> np.array:
    h, w = img.shape[:2]
    if not factor:
        factor = 300 / h
    img = cv2.resize(img, (int(factor * w), int(factor * h)))
    return img


def imageConcatenate(*imgs: np.array) -> np.array:
    shape = list()
    num_chanels = list()
    img_final = list()
    for img in imgs:
        shape.append(img.shape[:2])
        num_chanels.append(len(img.shape))
    shape = np.array(shape)
    num_chanels = np.array(num_chanels)
    h, w = np.max(shape, axis=0)
    c = np.max(num_chanels, axis=0)
    for img in imgs:
        if len(img.shape) == c:
            img_final.append(img)
        elif np.unique(img).shape[0] < 3:
            img_final.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return cv2.hconcat(img_final)


def lineStart(img: np.array) -> np.array:
    h, w = img.shape[:2]
    histogram = np.sum(img[h // 2:, :], axis=0)
    left_x = np.argmax(histogram[:w // 2])
    right_x = np.argmax(histogram[w // 2:]) + w // 2
    return left_x, right_x


def lineContinue(img: np.array, x_offset: int) -> int:
    histogram = np.sum(img, axis=0)
    w = histogram.shape[0]
    try:
        x_max = np.argmax(histogram) + x_offset
    except ValueError:
        x_max = x_offset
    return x_max


def indexPts(img: np.array, x_offset: int, y_offset: int):
    pts = np.argwhere(img == 255)
    if pts.size:
        pts[:, 0] = y_offset + pts[:, 0]
        pts[:, 1] = x_offset + pts[:, 1]
    return pts


def rectangleZone(img: np.array, left_x: int, right_x: int) -> None:
    # half box width
    W_BOX = 80
    # box height
    H_BOX = 40
    h, w = img.shape[:2]
    left_pts, right_pts = list(), list()
    pts = None
    l_pts, r_pts = None, None
    x_min_l, x_max_l, x_min_r, x_max_r = w, 0, w, 0
    for i in range(4):
        y_min = h - H_BOX - 1
        y_max = h - 1
        l_x_min = np.max([0, left_x - W_BOX // 2])
        l_x_max = np.min([w - 1, left_x + W_BOX // 2])
        r_x_min = np.max([0, right_x - W_BOX // 2])
        r_x_max = np.min([w - 1, right_x + W_BOX // 2])
        print('boxes:', l_x_min, l_x_max, r_x_min, r_x_max)

        l_offset, r_offset = left_x, right_x
        print('offset:', l_offset, r_offset)

        l_img = img[y_min:y_max, l_x_min:l_x_max]
        left_x = lineContinue(l_img, l_x_min)
        r_img = img[y_min:y_max, r_x_min:r_x_max]
        right_x = lineContinue(r_img, r_x_min)

        left = [(l_x_min, y_min), (l_x_max, y_max)]
        right = [(r_x_min, y_min), (r_x_max, y_max)]

        left_pts.append(left)
        right_pts.append(right)

        l_idx = indexPts(l_img, l_x_min, y_min)
        r_idx = indexPts(r_img, r_x_min, y_min)

        if isinstance(l_pts, np.ndarray):
            l_pts = np.append(l_pts, l_idx, axis=0)
        else:
            l_pts = l_idx

        if isinstance(r_pts, np.ndarray):
            r_pts = np.append(r_pts, r_idx, axis=0)
        else:
            r_pts = r_idx

        h -= H_BOX

        if x_min_l > l_x_min:
            x_min_l = l_x_min
        if x_min_r > r_x_min:
            x_min_r = r_x_min

        if x_max_l < l_x_max:
            x_max_l = l_x_max
        if x_max_r < r_x_max:
            x_max_r = r_x_max

    if l_pts.size:
        x_max = np.amax(left_pts, axis=0)[0]

        l_m2, l_m, l_b = np.polyfit(l_pts[:, 0], l_pts[:, 1], 2)
        poly = np.linspace(x_min_l, x_max_l, x_max_l - x_min_l // 2)

        y = (l_m2 * poly ** 2 + l_m * poly + l_b).astype(int)
        left_fit = np.zeros((poly.size, 1, 2))
        left_fit[:, 0, 1] = y
        left_fit[:, 0, 0] = poly
        left_fit = left_fit.astype(np.int32)

    else:
        left_fit = []
    if r_pts.size:
        r_m2, r_m, r_b = np.polyfit(r_pts[:, 1], r_pts[:, 0], 2)
        poly = np.linspace(x_min_r, x_max_r, x_max_r - x_min_r // 2)
        y = (r_m2 * poly ** 2 + r_m * poly + r_b).astype(int)
        right_fit = np.zeros((poly.size, 1, 2))
        right_fit[:, 0, 1] = y
        right_fit[:, 0, 0] = poly
        right_fit = right_fit.astype(np.int32)
    else:
        right_fit = []
    return left_pts, right_pts, l_pts, r_pts, left_fit, right_fit
