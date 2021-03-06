import cv2
import numpy as np
import sys
import argparse
from utils import hsv2cvhsv, availableFiles, check, resize, imageConcatenate, lineStart, lineContinue, indexPts, rectangleZone


def main():
    parser = argparse.ArgumentParser(description='Lane detection script')
    parser.add_argument('--video', default='data/Drive.mov',
                        type=str, help='Image path to detect lines')
    args = parser.parse_args()
    if args.video:
        file = availableFiles(args.video)
    if not file:
        error = ' '.join(['Could not find file:', args.path])
        sys.exit(error)

    print('Image file: ', file)
    # setup window size and placement
    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)
    # yellow lane upper and lower bound
    lower_yellow = hsv2cvhsv(np.array([35, 40, 50]))
    upper_yellow = hsv2cvhsv(np.array([55, 100, 100]))
    # white lane upper and lower bound
    lower_white = hsv2cvhsv(np.array([0, 0, 90]))
    upper_white = hsv2cvhsv(np.array([359, 5, 100]))
    # open capture video
    video = cv2.VideoCapture(file)
    print('Press Q to quit')
    # initialize ROI
    reg_interest = list()
    ret, frame = video.read()
    if ret:
        frame = resize(frame)
        h, w = frame.shape[:2]
        src = np.array([
            [6 * w // 13, 3 * h // 5], [7 * w // 13, 3 * h // 5],
            [1 * w // 13, h], [12 * w // 13, h]],
            dtype=np.float32)
        dst = np.array([
            [1 * w // 15, 0], [14 * w // 15, 0],
            [1 * w // 15, h], [14 * w // 15, h]],
            dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        roi_x = w // 15

    while video.isOpened():
        ret, frame = video.read()
        if check() or frame is None:
            break

        frame = resize(frame)
        warp = cv2.warpPerspective(frame, M, (w, h))
        blur = cv2.GaussianBlur(warp, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.add(mask_white, mask_yellow)
        edges = cv2.Canny(gray, 100, 200)
        warp_2 = cv2.warpPerspective(edges, M, (w, h))
        output = cv2.add(mask, edges)
        output[:, :roi_x] = 0
        output[:, -roi_x:] = 0
        output[:roi_x, :] = 0

        left_x, right_x = lineStart(output)

        left_pts, right_pts, left_idx, right_idx, left_fit, right_fit = rectangleZone(
            output, left_x, right_x)

        for left, right in zip(left_pts, right_pts):
            cv2.rectangle(blur, left[0], left[1], (255, 0, 0), 2)
            cv2.rectangle(blur, right[0], right[1], (0, 255, 0), 2)

        # cv2.polylines(blur, left_fit, True, (0, 0, 0), 10)
        # cv2.polylines(blur, right_fit, True, (0, 0, 0), 10)

        blur[right_idx[:, 0], right_idx[:, 1]] = (0, 0, 255)
        blur[left_idx[:, 0], left_idx[:, 1]] = (0, 0, 255)

        cv2.line(frame, (int(src[0, 0]), int(src[0, 1])),
                 (int(src[2, 0]), int(src[2, 1])), (0, 0, 255), 2)
        cv2.line(frame, (int(src[1, 0]), int(src[1, 1])),
                 (int(src[3, 0]), int(src[3, 1])), (0, 0, 255), 2)
        cv2.line(blur, (int(dst[0, 0]), int(dst[0, 1])),
                 (int(dst[2, 0]), int(dst[2, 1])), (0, 0, 255), 2)
        cv2.line(blur, (int(dst[1, 0]), int(dst[1, 1])),
                 (int(dst[3, 0]), int(dst[3, 1])), (0, 0, 255), 2)

        img = imageConcatenate(blur, edges)
        cv2.imshow(win_name, img)

        # cv2.waitKey(0)
        # break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
