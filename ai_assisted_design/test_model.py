import numpy as np
from classes.models import alexnet_v2
from classes.camera import Camera
import classes.effects as fx
import classes.imgops as imgops
import classes.web as web
import cv2
import os
import sys


WIDTH = 250  # Image Width
HEIGHT = 250  # Image Height
LR = 1e-3  # Learning Rate
EPOCHS = 8


# Load model
model = alexnet_v2(WIDTH, HEIGHT, LR, output=4)
MODEL_NAME = 'pywebdevelopment_{0}_{1}_{2}_epochs.model'.format(LR, 'alexnet_v2', EPOCHS)
print('Model Name: {0}'.format(MODEL_NAME))
model.load('{0}/model/{1}'.format(os.getcwd(), MODEL_NAME))

print('LOADED PREVIOUS MODEL...')


def main():

    camera = Camera()
    current_frame = []

    while True:
        test_view = camera.get_camera(mirror=False)
        # Generate image contours and visualize
        img_contours = fx.get_contours(test_view, max_contours=10, num_sides=4, debug=False)[1]
        try:
            img_contours = imgops.sort_contours(img_contours, method='top-to-bottom')[0]
        except:
            pass
        cv2.imshow('Selected Contours', fx.visualize_contours(test_view.copy(), img_contours))
        img_isolated = imgops.crop_image_by_contour(test_view, img_contours, perspective_transform=True)
        key_input = cv2.waitKey(1) & 0xFF
        if key_input == ord('g'):
            # Loop through each isolated image and predict classification
            for img in img_isolated:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)[1]
                img = cv2.dilate(img, np.ones((3, 3), 'uint8'), iterations=1)
                img = cv2.resize(img, (250, 250))

                prediction = model.predict([img.reshape(WIDTH, HEIGHT, 1)])[0]
                print(prediction)
                prediction_choice = np.argmax(prediction) + 1
                print(prediction_choice)

                if prediction_choice == 1:
                    current_frame.append('module_header')
                elif prediction_choice == 2:
                    current_frame.append('module_item')
                elif prediction_choice == 3:
                    current_frame.append('module_content')
                elif prediction_choice == 4:
                    current_frame.append('module_create_item')

            for classification in current_frame:
                print('%s' % classification)

            print('-'*20)
            web.create_template('WebView', current_frame)
            # cv2.destroyWindow('Selected Contours')
            current_frame = []

        elif key_input == ord('c'):
            web.clear()
        elif key_input == ord('q'):
            web.clear()
            sys.exit()


if __name__ == '__main__':
    main()
