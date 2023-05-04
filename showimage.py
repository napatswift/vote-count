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


def main():
  """
  The main function of the code.

  """

  # Load the JSON file.
  with open('output/test/train.json') as f:
    data = json.load(f)

  # Select 15 random images from the list.
  data_list = list(data['data_list'])
  for i in range(15):
    data = random.choice(data_list)
    data_list.remove(data)

    # Read the image.
    img = cv2.imread('output/test/imgs/' + data['img_path'])
    if img is None:
      raise FileExistsError('image file not found')

    # Draw the bounding boxes of the text regions in the image.
    for text_reg in data['instances']:
      polygon = text_reg['polygon']
      polygon = [int(x) for x in polygon]
      for i in range(len(polygon) // 2 - 1):
        ix0 = i * 2
        ix1 = (i + 1) * 2
        cv2.line(img, [polygon[ix0], polygon[ix0 + 1]], [polygon[ix1], polygon[ix1 + 1]], (255, 0, 0), 2)
      bbox = text_reg['bbox']
      bbox = [int(x) for x in bbox]
      cv2.rectangle(img, bbox[:2], bbox[2:], (0, 0, 255), 1)

    # Display the image.
    cv2.imshow(data['img_path'], img)
    cv2.waitKey(500)


if __name__ == '__main__':
  main()
