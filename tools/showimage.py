"""
This code shows a random image from a dataset and draws the bounding boxes of the text regions in the image.

Usage:

python showimage.py

The code will first load the JSON file `output/test/train.json`. This file contains a list of images and their associated text regions. The code will then randomly select 15 images from the list and display them, one at a time. For each image, the code will draw the bounding boxes of the text regions in the image.

The code uses the following libraries:

* `json` - for loading the JSON file
* `cv2` - for reading and displaying images
* `random` - for selecting random images from the list

"""

import json
import cv2
import random
import argparse
import os.path as osp


def main():
    """
    The main function of the code.

    """

    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='train.json',
                        help='the path to the JSON file')
    args = parser.parse_args()
    data_path = args.data_path
    data_dir = osp.dirname(data_path)

    # Load the JSON file.
    with open(f'{data_path}') as f:
        data = json.load(f)

    # Select 15 random images from the list.
    data_list = list(data['data_list'])
    for i in range(15):
        data = random.choice(data_list)
        data_list.remove(data)

        # Read the image.
        img = cv2.imread(f'{data_dir}/imgs/' + data['img_path'])
        if img is None:
            raise FileExistsError('image file not found')

        # Draw the bounding boxes of the text regions in the image.
        for text_reg in data['instances']:
            # polygon = text_reg['polygon']
            # polygon = [int(x) for x in polygon]
            # for i in range(len(polygon) // 2 - 1):
            #     ix0 = i * 2
            #     ix1 = (i + 1) * 2
            #     cv2.line(img, [polygon[ix0], polygon[ix0 + 1]],
            #              [polygon[ix1], polygon[ix1 + 1]], (255, 0, 0), 2)
            bbox = text_reg['bbox']
            bbox = [int(x) for x in bbox]
            cv2.rectangle(img, bbox[:2], bbox[2:], (255, 0, 0), 1)

        # Display the image.
        cv2.imshow('Image', img)

        key = cv2.waitKey(0)
        # if key is escape, exit
        if key == 27:
            break
        # if key is 's', save the image
        elif key == ord('s'):
            cv2.imwrite('image.png', img)


if __name__ == '__main__':
    main()
