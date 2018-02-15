import cv2
import numpy as np


def stroke_edges(src, blur_size=3, edge_size=5, multiplier=1):
    if blur_size >= 3:
        blurred_src = cv2.GaussianBlur(src, (blur_size, blur_size), 0)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8UC4, gray_src, ksize=edge_size, scale=multiplier)
    gray_src = cv2.bilateralFilter(gray_src, 22, 75, 75)
    normalized_inv_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inv_alpha
    output_img = cv2.merge(channels)
    return output_img


def find_edges(src, kblur=1, thresh1=100, thresh2=200, denoise=False):
    output = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    if denoise:
        output = cv2.fastNlMeansDenoising(output, None, 10, 7, 18)
    output = cv2.Canny(output, threshold1=thresh1, threshold2=thresh2)
    output = cv2.blur(output, (kblur, kblur))

    return output


def get_contours(src, sort_reverse=False, max_contours=10, num_sides=0, debug=False):
    output = find_edges(src, kblur=3, thresh1=30, thresh2=120, denoise=True)
    output = cv2.dilate(output, np.ones((5, 5), 'uint8'), iterations=2)
    # output = cv2.threshold(output, 82, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=sort_reverse)[:max_contours]
    if num_sides > 0:
        contours_sides = get_contours_by_sides(contours, sides=num_sides)
    else:
        contours_sides = None
    if debug:
        cv2.imshow('contour -debug', output)
    return contours, contours_sides


def get_contours_by_sides(contours, sides=4):
    if sides > 0:
        screen_contours = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            if len(approx) == sides:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (w, h) = cv2.boundingRect(approx)[2:]
                ar = w / float(h)
                if 0.5 <= ar:
                    screen_contours.append(approx)
    else:
        screen_contours = None

    return screen_contours


def visualize_contours(src_img, contours, color=(0, 255, 0), sides=4):
    screen_contours = get_contours_by_sides(contours, sides)
    cv2.drawContours(src_img, screen_contours, -1, color, 2)
    return src_img
