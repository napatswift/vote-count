import json
import os

import cv2


class FileSaver:
    def __init__(self, image_dir, localization_dir):
        pass

    def save(self, image, bboxes, segmap=None,):
        pass

    def clean(self, json_path):
        pass


class TextFileSaver(FileSaver):
    def __init__(self, image_dir, localization_dir):
        self.image_dir = image_dir

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.localization_dir = localization_dir

        if not os.path.exists(self.localization_dir):
            os.makedirs(self.localization_dir)

        self.image_id = 0
        self.data_list = []

    def save(self, image, bboxes, segmap=None,):
        image_name = f'img_{self.image_id}'
        cv2.imwrite(os.path.join(self.image_dir, image_name+'.jpg'), image)
        cv2.imwrite(os.path.join(self.image_dir, image_name +
                    '_mask'+'.png'), segmap.get_arr())
        # cv2.imwrite(os.path.join(image_dir, image_name+'_keypoint'+'.jpg'), aug_kp.draw_on_image(aug_image,size=10))

        fp = open(os.path.join(self.localization_dir,
                  'gt_'+image_name+'.txt'), 'w')
        for bbox in bboxes.bounding_boxes:
            for point in bbox.to_keypoints():
                fp.write(str(point.x_int)+',')
                fp.write(str(point.y_int)+',')

            fp.write(bbox.label)
            fp.write('\n')
        fp.close()

        self.data_list.append({
            'img_path': f'{image_name}.jpg',
            'height': image.shape[0],
            'width': image.shape[1],
            'instances': [
                {
                    'bbox': [bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int],
                    'text': bbox.label,
                    'bbox_label': 0,  # text
                    'ignore': False,
                } for bbox in bboxes.bounding_boxes
            ]
        })

        self.image_id += 1

    def clean(self, json_path):
        return


class MMOCRFileSaver(FileSaver):
    """https://mmocr.readthedocs.io/en/dev-1.x/api/generated/mmocr.datasets.OCRDataset.html#mmocr.datasets.OCRDataset"""

    def __init__(self, image_dir, localization_dir):
        self.image_dir = image_dir

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.localization_dir = localization_dir

        if not os.path.exists(self.localization_dir):
            os.makedirs(self.localization_dir)

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
