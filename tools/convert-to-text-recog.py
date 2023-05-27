import json
import numpy as np
import cv2
import os
from tqdm import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from random import randint, shuffle
import argparse
from pathlib import Path

IMG_ID = 0
AUG_SEQ = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Dropout(p=(.0, .0015))),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation(
        (0.5, 1.5), per_channel=True)),
])


def get_bbox_and_text(rect_text_list):
    """Get bbox and text from rect_text_list"""
    text_list = []
    bbox_list = []
    for rect_text in rect_text_list:
        if 'text' not in rect_text.keys():
            continue
        if not check_size(rect_text['bbox']):
            continue
        text_list.append(rect_text['text'])
        bbox_list.append(rect_text['bbox'])
    return bbox_list, text_list


def check_size(bbox):
    x0, y0, x1, y1 = bbox
    return (x1-x0) * (y1-y0) > 100


def save_image(filename: str, bbox: BoundingBox, img: np.array):
    assert isinstance(bbox, BoundingBox)
    assert isinstance(img, np.ndarray)
    # r = 250/bbox.width
    w = bbox.width  # * r
    h = bbox.height  # * r
    text_img = img[bbox.y1_int:bbox.y2_int, bbox.x1_int:bbox.x2_int, :]
    return cv2.imwrite(filename,
                       cv2.resize(text_img, (int(w), int(h))))


def get_bboxes_aug(bboxes, texts, shape):
    if randint(0, 100) < 50:
        bboxes = [(x0-randint(1, 8),
                   y0-randint(1, 8),
                   x1+randint(1, 8),
                   y1+randint(1, 8)) for x0, y0, x1, y1 in bboxes]
    return BoundingBoxesOnImage([BoundingBox(*b, t) for b, t in zip(bboxes, texts)], shape=shape)

class TextRecogDataset:
    """https://mmocr.readthedocs.io/en/dev-1.x/migration/dataset.html#id2"""

    def __init__(self) -> None:
        self.data_list = []
        self.labels_count = {}
        self.max_count = 100
    
    def add_data(self, image_path, text):
        text = str(text)
        self.labels_count[text] = self.labels_count.get(text, 0) + 1
        if self.labels_count[text] > self.max_count:
            return False
        
        self.data_list.append({
            'img_path': image_path,
            'instances': [dict(text=text)]
        })

        return True
    
    @property
    def image_id(self):
        return len(self.data_list)
    
    def save(self, fpath):
        with open(fpath, 'w') as fp:
            json.dump(dict(metainfo=dict(dataset_type='TextRecogDataset', task_name='textrecog'),
                           data_list=self.data_list), fp, ensure_ascii=False)

def cvt(dir_name, data_list, matadata_fpath):
    image_per_dir = 1000
    text_recog_dataset = TextRecogDataset()

    def _helper_save_image(img_dir, image, bboxes, text_recog_dataset):
        global image_id
        aug_image, aug_bboxes = AUG_SEQ(image=image, bounding_boxes=bboxes)
        aug_bboxes: BoundingBoxesOnImage = aug_bboxes.remove_out_of_image(True, True)
        for bbox in aug_bboxes.bounding_boxes:
            try:
                img_name = f'{text_recog_dataset.image_id}.jpeg'
                sub_dir = str(text_recog_dataset.image_id//image_per_dir)
                img_path = os.path.join(sub_dir, img_name)
                # Make dir
                os.makedirs(os.path.join(img_dir, sub_dir), exist_ok=True)
                # Increment image id
                if text_recog_dataset.add_data(img_path, bbox.label):
                    # Save image
                    save_image(os.path.join(img_dir, img_path), bbox, aug_image)
                    # Save metadata
                    fp_metadata.write(f'{img_path},{bbox.label}\n',)

            except Exception as e:
                print(e)
                pass

    def _convert_one_image(data, text_recog_dataset):
        # Get bbox and text
        bbox_list, text_list = get_bbox_and_text(data['instances'])
        # Read image
        image = cv2.imread(os.path.join(dir_name, 'imgs', data['img_path']))
        # Get bounding boxes
        bboxes_augimg = get_bboxes_aug(bbox_list, text_list, image.shape)
        # Save image
        for _ in range(2):
            _helper_save_image(img_dir, image, bboxes_augimg, text_recog_dataset)

    fp_metadata = open(matadata_fpath, 'a')
    img_dir = os.path.join(f'{dir_name}_recog', 'imgs',)
    image_per_dir = 1000

    try:
        for data in tqdm(data_list):
            _convert_one_image(data, text_recog_dataset)
    except KeyboardInterrupt:
        pass
    
    text_recog_dataset.save(os.path.join(img_dir, 'text_recog.json'))
    fp_metadata.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, )
    parser.add_argument('--output', type=str,
                        default='output', help='output dir')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    assert dataset_path.exists()
    assert dataset_path.is_file()

    dir_name = dataset_path.parent

    # load data
    data = json.load(open(args.dataset))

    img_dir = os.path.join(f'{dir_name}_recog', 'imgs',)
    os.makedirs(img_dir,)

    matadata_fpath = os.path.join(img_dir, 'metadata.csv')
    print(matadata_fpath)

    with open(matadata_fpath, 'w') as fp:
        fp.write('file_name,text\n')
    shuffle(data['data_list'])

    cvt(dir_name, data['data_list'], matadata_fpath)
