import json
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import random
import os
import argparse
from tqdm import tqdm
import augraphy as agh

# create a augmentation pipeline to mimick the real world data
seq = iaa.Sequential([
    # # Add perspective transformations to mimic skewed or rotated documents
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(
        scale=(0.01, 0.05), fit_output=True)),
    # # Perturb the colors by adjusting brightness, adding noise, etc.
    iaa.SomeOf((1, 3), [
        iaa.Multiply((0.5, 1)),
        iaa.GammaContrast(gamma=(0.5, 1)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    ]),
    # # Apply affine transformations (shearing, rotation, etc.)
    iaa.SomeOf((0, 2), [
        iaa.ShearX((-5, 5), fit_output=True, cval=(0, 255)),
        iaa.ShearY((-5, 5), fit_output=True, cval=(0, 255)),
        iaa.Rotate((-5, 5), fit_output=True, cval=(0, 255)),
        iaa.ElasticTransformation(alpha=1500, sigma=200, order=1),
        iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
            iaa.Sequential([iaa.MotionBlur(k=(3, 7)),
                            iaa.GammaContrast(gamma=(.1, 3))])
        ),
    ]),
    iaa.BlendAlphaMask(
        iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
        iaa.MotionBlur(k=(3, 7))
    ),
])


def augment_image(image):
    return agh.AugmentationSequence([
        agh.Gamma(gamma_range=(0.1, 2), p=0.8),
        agh.LowInkPeriodicLines(p=0.1),
        agh.BleedThrough(intensity_range=(0.1, 0.3), p=0.8),
        agh.DirtyDrum(line_concentration=0.05, line_width_range=(1, 2), p=0.1),
    ])(image)[0]


class MMOCRFileSaver:
    """https://mmocr.readthedocs.io/en/dev-1.x/api/generated/mmocr.datasets.OCRDataset.html#mmocr.datasets.OCRDataset"""

    def __init__(self, image_dir,):
        self.image_dir = image_dir

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.image_id = 0
        self.data_list = []

    def save(self, image, bboxes, segmap=None,):
        image_name = f'img_{self.image_id}'
        cv2.imwrite(os.path.join(self.image_dir, image_name+'.jpg'), image)
        padding = 1
        if segmap is not None:
            cv2.imwrite(os.path.join(self.image_dir,
                                     image_name + '_mask'+'.png'), segmap.get_arr())

        text_instances = [
            {
                'bbox': [bbox.x1_int-padding, bbox.y1_int-padding,
                         bbox.x2_int+padding, bbox.y2_int+padding],
                'polygon': self._get_polygon(bbox.to_keypoints()),
                'text': bbox.label,
                'bbox_label': 0,  # text
                'ignore': False,
            } for bbox in bboxes.bounding_boxes
        ]

        self.data_list.append({
            'img_path': f'{image_name}.jpg',
            'height': image.shape[0],
            'width': image.shape[1],
            'instances': text_instances
        })

        self.image_id += 1

    def _get_polygon(self, keypoints):
        """Get polygon from keypoints."""

        points = []
        for point in keypoints:
            points.append(point.x_int)
            points.append(point.y_int)
        assert len(points) == 8
        return points

    def clean(self, json_path):
        with open(json_path, 'w') as fp:
            json.dump(dict(metainfo=dict(dataset_type='TextDetDataset',
                                         task_name='textdet',
                                         category=[dict(id=0, name='text')]),
                           data_list=self.data_list), fp, ensure_ascii=False,)


def augment_data(data_dir, data_instace, file_saver):
    image_prefix = 'imgs'
    image_path = data_instace['img_path']
    image = cv2.imread(os.path.join(data_dir, image_prefix, image_path))

    if image is None:
        raise FileExistsError('image file not found')

    bounding_boxes = []
    for instance in data_instace['instances']:
        bbox = instance['bbox']
        bounding_boxes.append(BoundingBox(
            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))

    bbs = BoundingBoxesOnImage(bounding_boxes, shape=image.shape)

    num_aug = random.randint(1, 4)
    # Augment BBs and images.
    for _ in range(num_aug):
        image_aug, bbs_aug = seq(
            image=augment_image(image), bounding_boxes=bbs)
        file_saver.save(image_aug, bbs_aug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,)
    args = parser.parse_args()

    data_path = args.data_path
    data_file_name = os.path.basename(data_path).split('.')[0]
    data_dir = os.path.dirname(data_path)

    # Load the JSON file.
    with open(f'{data_path}') as f:
        data = json.load(f)

    data_list = list(data['data_list'])

    file_saver = MMOCRFileSaver(os.path.join(data_dir, 'imgs'),)
    file_saver.data_list.extend(data_list)
    try:
        for data_instace in tqdm(data_list):
            augment_data(data_dir, data_instace, file_saver)
            # file_saver.save(image, bbs)
    except KeyboardInterrupt:
        pass

    file_saver.clean(os.path.join(data_dir, data_file_name + '_aug.json'))


if __name__ == '__main__':
    main()
