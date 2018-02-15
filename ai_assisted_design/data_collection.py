import numpy as np
import cv2
import os
import random

from classes.camera import Camera
import classes.effects as fx
import classes.imgops as imgops


def get_starting_file(file_name):
    if file_name.endswith('.npy'):
        file_name.replace('.npy', '')
    starting_value = 1
    while True:
        ammended_file_name = '{0}_{1}.npy'.format(file_name, starting_value)
        if os.path.isfile('{0}/training_data/{1}'.format(os.getcwd(), ammended_file_name)):
            print('{0} exists! Checking for next file...'.format(ammended_file_name))
            starting_value += 1
        else:
            print('{0} does not exist! Starting from here...'.format(ammended_file_name))
            return ammended_file_name, starting_value


def save_data(file_name, data, shuffle=False):
    if shuffle:
        data = random.shuffle(data)
    file_output = '{0}/training_data/{1}'.format(os.getcwd(), file_name)
    np.save(file_output, data)
    print('Saved {0}'.format(file_name))


def collect_data(file_name, max_file_data=10):
    # Establish empty data array and camera
    print('Max file data: %s entries' % max_file_data)
    camera = Camera()
    training_data = []

    output_name, starting_value = get_starting_file(file_name)
    paused = False

    main_view = None

    while True:
        main_view = camera.get_camera(mirror=False)
        # Generate image contours and visualize
        img_contours = fx.get_contours(main_view, max_contours=50, num_sides=4, debug=False)[1]
        cv2.imshow('Selected Contours', fx.visualize_contours(main_view.copy(), img_contours))
        img_isolated = imgops.crop_image_by_contour(main_view, img_contours, perspective_transform=True)
        if cv2.waitKey(1) & 0xFF == ord('c') and not paused:

            # Loop through each isolated image and manually discern classification
            for img in img_isolated:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)[1]
                img = cv2.dilate(img, np.ones((3, 3), 'uint8'), iterations=1)
                cv2.imshow('Choose Classification', cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))
                img = cv2.resize(img, (250, 250))
                classification_input = cv2.waitKey(0) & 0xFF
                if classification_input == ord('1'):
                    img_classification = 'module_header'
                    # Assign Training data
                    training_data.append([img, img_classification])
                    cv2.destroyWindow('Choose Classification')
                if classification_input == ord('2'):
                    img_classification = 'module_item'
                    # Assign Training data
                    training_data.append([img, img_classification])
                    cv2.destroyWindow('Choose Classification')
                if classification_input == ord('3'):
                    img_classification = 'module_content'
                    # Assign Training data
                    training_data.append([img, img_classification])
                    cv2.destroyWindow('Choose Classification')
                if classification_input == ord('4'):
                    img_classification = 'module_create_item'
                    # Assign Training data
                    training_data.append([img, img_classification])
                    cv2.destroyWindow('Choose Classification')
                if classification_input == ord('0'):
                    # Not a valid image. Pass.
                    cv2.destroyWindow('Choose Classification')

                img_classification = None

                if len(training_data) % 10 == 0:
                    print(len(training_data))

                    if len(training_data) == max_file_data:
                        save_data(output_name, training_data)
                        training_data = []
                        break
                        starting_value += 1
            print('Press "C" to continue...')
            cv2.destroyWindow('Selected Contours')


if __name__ == '__main__':
    collect_data('training_data', max_file_data=50)
