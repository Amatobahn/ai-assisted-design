from classes.screen import *
from classes.camera import *
import classes.effects as fx
import classes.imgops as imgops


if __name__ == '__main__':
    screen = Screen()
    camera = Camera()

    while True:
        # img = screen.grab_screen(region=(0, 100, 800, 600), preprocess=True, stroke=True)
        # img_pre = screen.grab_screen(region=(0, 100, 800, 600), preprocess=True, stroke=False)
        orig_img = camera.get_camera(mirror=False)
        # cam_img = fx.stroke_edges(orig_img, blur_size=5, edge_size=5, multiplier=1.5)
        cam_img = fx.find_edges(orig_img, kblur=1, thresh1=80, thresh2=120)
        contours = fx.get_contours(orig_img, max_contours=25, num_sides=4, debug=False)[1]
        contour_img = fx.visualize_contours(orig_img.copy(), contours, sides=4)
        cropped_images = imgops.crop_image_by_contour(orig_img, contours, show_images=False, img_scale=2)
        cv2.imshow('WebCam', contour_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
