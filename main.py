from curses import window
from matplotlib import image
from utils import imageConcatenate, lineStart, lineContinue, indexPts, rectangleZone
import cv2
import numpy as np
import argparse
from image import Image


def main():
    parser = argparse.ArgumentParser(description='Lane detection script')
    parser.add_argument('--path', default='data/Drive.mov',
                        type=str, help='Video path to detect lines')
    args = parser.parse_args()
    image = Image(args.path)
    window_name = 'Frame'
    image.set_window(window_name, 20, 20)

    # yellow lane upper and lower bound
    lower_yellow = image.hsv_to_cv_hsv(np.array([35, 40, 50]))
    upper_yellow = image.hsv_to_cv_hsv(np.array([55, 100, 100]))
    # white lane upper and lower bound
    lower_white = image.hsv_to_cv_hsv(np.array([0, 0, 90]))
    upper_white = image.hsv_to_cv_hsv(np.array([359, 5, 100]))
    # open capture video
    video, width, height = image.video_capure()
    # refactor image width and height
    width, height = image.factor((width, height))

    roi_vertices = np.array(
        [[[0, height],
          [width // 2, height // 2],
            [width, height]]], dtype=np.int32)
    mask = image.mask_region_of_interest((width, height), roi_vertices)

    print('Press Q to quit')
    while video.isOpened():
        ret, frame = video.read()
        if image.check_key() or frame is None:
            break
        frame = image.resize(frame, (width, height))

        gray = image.gray(frame)
        edges = image.canny_edge_detector(gray, 100, 200)
        roi = image.bitwise_and(edges, mask)
        lines = image.hough_lines(roi, 6, np.pi / 60, 160, 40, 25)
        lines = image.filter_slope(lines)

        image.draw_lines(frame, lines)
        # print(lines)

        # img = imageConcatenate(blur, edges)
        cv2.imshow(window_name, frame)

        # cv2.waitKey(0)
        # break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
