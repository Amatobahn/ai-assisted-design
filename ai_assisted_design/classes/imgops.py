# import the necessary packages
import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_output = cv2.warpPerspective(image, M, (max_width, max_height))

    # return the warped image
    return warped_output


def sort_contours(contours, method='top-to-bottom'):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # Handle if we need to sort in reverse
    if method == 'bottom-to-top':
        reverse = True

    # Handle if we are sorting against the y-axis rather
    #  than the x-axis of the bounding boxes
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1

    # Construct the list of bounding boxes and sort them from top to bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                             key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return contours, bounding_boxes


def crop_image_by_contour(original_image, contours, perspective_transform=False, show_images=False, img_scale=1):
    output_img = []

    for contour_set in contours:
        if perspective_transform:
            corrected_img = four_point_transform(original_image, contour_set.reshape(4, 2))
        else:
            mask = np.zeros_like(original_image)
            cv2.drawContours(mask, contour_set, -1, 255, -1)
            corrected_img = np.zeros_like(original_image)
            corrected_img[mask == 255] = original_image[mask == 255]
        output_img.append(corrected_img)
    if show_images:
        for i in range(0, len(output_img)):
            (h, w) = output_img[i].shape[:2]
            resize_output = cv2.resize(output_img[i], (w * img_scale, h * img_scale))
            cv2.imshow('output {0}'.format(i), resize_output)
    return output_img
