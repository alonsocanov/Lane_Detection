# Hough Transform
import cv2
import numpy as np
import os
import sys
import glob
import argparse


def get_slope(lines):
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
    return slopes


def get_intecept(lines, slopes):
    intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (
        lines[:, 2] + lines[:, 0])) / 2
    return intercepts


def showImg(img: np.array) -> None:
    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# check available files in path
def availableFiles(path) -> list:
    full_path = glob.glob(path, recursive=True)
    if len(full_path):
        file = full_path[0]
    else:
        error = ' '.join(['Could not find file:', path])
        sys.exit(error)
    return file


# color normalization of HSV to OpenCV HSV
def hsv2cvhsv(hsv: np.array) -> np.array:
    # For HSV, Hue range is [0,179], Saturation range is [0,255]
    # and Value range is [0,255]. Different software use different scales.
    # So if you are comparing in OpenCV values with them, you need to normalize these ranges.
    hsv_cv = np.array([179, 255, 255])
    hsv_orig = np.array([360, 100, 100])
    cv_hsv = np.divide((hsv * hsv_cv), hsv_orig)
    return cv_hsv


def main():
    parser = argparse.ArgumentParser(description='Line traking')
    parser.add_argument('path', type=str, help='Image path to detect lines')
    args = parser.parse_args()
    file = availableFiles(args.path)
    if not file:
        error = ' '.join(['Could not find file:', path])
        sys.exit(error)
    print('Image file: ', file)

    yellow_lower = hsv2cvhsv(np.array([37, 75, 75]))
    yellow_upper = hsv2cvhsv(np.array([50, 100, 100]))

    # read original image file declared on argument
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.blur(gray, (5, 5))
    # mask = cv2.inRange(blur, yellow_lower, yellow_upper)

    # erode = cv2.erode(mask, (5, 5), iterations=1)
    # combination = cv2.add(mask, gray)
    edges = cv2.Canny(blur, 10, 150)

    # Hough line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=40)
    lines = np.squeeze(lines)

    # draw lines detected on original image
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    max_slope_threshold = 0.7

    slopes = get_slope(lines)
    intercepts = get_intecept(lines, slopes)

    bool_lines = (np.abs(slopes) > min_slope_threshold) * (np.abs(slopes) < max_slope_threshold)
    lines = lines[bool_lines]
    intercepts = intercepts[bool_lines]
    slopes = slopes[bool_lines]
    print(slopes, '\n', lines)
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    showImg(img)
    sys.exit()

    merged_lines = []
    examined_lines = []
    for slope, intercept in zip(slopes, intercepts):
        slope_diff = np.abs(slopes - slope)
        intercept_diff = np.abs(intercepts - intercept)
        vals = np.array(slope_diff < slope_similarity_threshold) * np.array(intercept_diff < intercept_similarity_threshold)
        # efficiency
        merged_lines.append(lines[vals])
        examined_lines.append(lines[vals])
    # Step 4: Merge all lines in clusters using mean averaging
    temp = []
    for cluster in merged_lines:
        temp.append(np.average(cluster, axis=0))
    temp = np.array(temp)
    merged_lines = np.unique(temp, axis=0)

    merged_lines = np.array(merged_lines, dtype=np.int32)
    for x1, y1, x2, y2 in merged_lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    showImg(img)


if __name__ == "__main__":
    main()
