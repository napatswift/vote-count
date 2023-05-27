from PIL import Image, ImageFont, ImageDraw
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import augraphy as agh
from faker import Faker
import random
import attacut
import re
import pandas as pd
import os
from tqdm import trange
import numpy as np
import cv2
import json
import argparse

last_names_th = open('data/family_names_th.txt').read().split('\n')[:-1]
names_th = open('data/female_names_th.txt').read().split('\n')[:-1]
geo_names = pd.read_csv('data/tambons.csv')
months_th = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
             'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
political_parties_th = ['ประชาธิปัตย์', 'ประชากรไทย', 'ความหวังใหม่', 'เครือข่ายชาวนาแห่งประเทศไทย', 'เพื่อไทย', 'เพื่อแผ่นดิน', 'ชาติพัฒนา', 'ชาติไทยพัฒนา', 'อนาคตไทย', 'ภูมิใจไทย', 'สังคมประชาธิปไตยไทย', 'ประชาสามัคคี', 'ประชาธิปไตยใหม่', 'พลังชล', 'ครูไทยเพื่อประชาชน', 'พลังสหกรณ์', 'พลังท้องถิ่นไท', 'ถิ่นกาขาวชาววิไล',
                        'รักษ์ผืนป่าประเทศไทย', 'ไทรักธรรม', 'เสรีรวมไทย', 'รักษ์ธรรม', 'เพื่อชาติ', 'พลังประชาธิปไตย', 'ภราดรภาพ', 'พลังไทยรักชาติ', 'เพื่อชีวิตใหม่', 'ก้าวไกล', 'ทางเลือกใหม่', 'ประชาภิวัฒน์', 'พลเมืองไทย', 'พลังไทยนำไทย', 'พลังธรรมใหม่', 'ไทยธรรม', 'ไทยศรีวิไลย์', 'รวมพลังประชาชาติไทย', 'สยามพัฒนา', 'เพื่อคนไทย', 'พลังปวงชนไทย', 'พลังไทยรักไทย']

# create a augmentation pipeline to mimick the real world data
seq = iaa.Sequential([
    # # Add perspective transformations to mimic skewed or rotated documents
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(
        scale=(0.01, 0.05), fit_output=True)),
    # # Perturb the colors by adjusting brightness, adding noise, etc.
    iaa.SomeOf((1, 3), [
        iaa.Multiply((0.3, 1.1)),
        iaa.GammaContrast(gamma=(0.5, 1)),
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
    iaa.ElasticTransformation(alpha=2000, sigma=160, ),
])


def augment_image(image):
    return agh.AugmentationSequence([
        agh.Gamma(gamma_range=(0.1, 1.5), p=0.8),
        agh.LowInkPeriodicLines(p=0.3),
        agh.BleedThrough(intensity_range=(0.1, 0.3), p=0.8),
        agh.DirtyDrum(p=0.3),
    ])(image)[0]


class BBox:
    def __init__(self, x0=None, y0=None, x1=None, y1=None):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def merge(self, x0, y0, x1, y1):
        if self.x0 is None or self.y0 is None or self.x1 is None or self.y1 is None:
            self.x0 = x0
            self.x1 = x1
            self.y0 = y0
            self.y1 = y1
        else:
            self.x0 = min(self.x0, x0)
            self.y0 = min(self.y0, y0)
            self.x1 = max(self.x1, x1)
            self.y1 = max(self.y1, y1)

    def to_list(self,):
        return [self.x0, self.y0, self.x1, self.y1]

    def __repr__(self,):
        return f'({self.x0},{self.y0},{self.x1},{self.y1})'


class RandomFont:
    def __init__(self, font_dir: str):
        self.font_paths = [os.path.join(font_dir, fp)for fp in os.listdir(
            font_dir) if fp.endswith('.ttf')]

    def get(self, size=20):
        """
        Gets a random font with the specified size.

        Args:
            size: The size of the font.

        Returns:
            A `PIL.ImageFont` object representing the random font.
        """
        font_path = random.choice(self.font_paths)
        default_font_size = 10
        font = ImageFont.truetype(font_path, default_font_size)
        test_text = 'นี่คือข้อความใช้สำหรับทดสอบ'
        text_bbox = font.getbbox(test_text)
        height = text_bbox[3] - text_bbox[1]
        return ImageFont.truetype(font_path, size=int(size*default_font_size/height+0.5))


def arabic2th(n):
    return chr(ord(n)+(ord('๑')-ord('1')))


def unit_process(val):
    # https://suilad.wordpress.com/2012/05/09/code-python-%E0%B9%81%E0%B8%9B%E0%B8%A5%E0%B8%87%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B9%80%E0%B8%A5%E0%B8%82-%E0%B9%84%E0%B8%9B%E0%B9%80%E0%B8%9B%E0%B9%87%E0%B8%99-%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%AB/
    thai_number = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่",
                   "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
    unit = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน")
    length = len(val) > 1
    result = ''

    for index, current in enumerate(map(int, val)):
        if current:
            if index:
                result = unit[index] + result

            if length and current == 1 and index == 0:
                result += 'เอ็ด'
            elif index == 1 and current == 2:
                result = 'ยี่' + result
            elif index != 1 or current != 1:
                result = thai_number[current] + result

    return result


def thai_num2text(number):
    s_number = str(number)[::-1]
    n_list = [s_number[i:i + 6].rstrip("0")
              for i in range(0, len(s_number), 6)]
    result = unit_process(n_list.pop(0))

    for i in n_list:
        result = unit_process(i) + 'ล้าน' + result

    return result


fonts = {'sarabun': RandomFont('fonts/formal'),
         'handwriting': RandomFont('fonts/handwriting'), }

image_width = 826
image_height = 1169


class TextTemplate:
    def __init__(self) -> None:
        self.template_files = [os.path.join('templates', f) for f in os.listdir(
            'templates') if f.endswith('.txt')]
        random.shuffle(self.template_files)
        self.current_idx = 0

    def gen(self,):
        template_file = self.template_files[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.template_files)
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()

        return self._get_tokens(template)

    def _get_tokens(self, text_template):
        big_pieces = re.split(
            '(\<[^\>]*\>|\t|\n| |\{[^\}]*\}|%%\w+|\|)', text_template)
        tokens = []

        for piece in big_pieces:
            if re.findall('[\u0E00-\u0E0F]+', piece):
                tokens.extend(attacut.tokenize(piece))
            elif piece == '':
                continue
            else:
                tokens.append(piece)
        return tokens


def _main(file_saver, text_template_generator):
    # create blank white paper
    image = Image.new('RGB', (image_width, image_height,), '#fff')
    draw = ImageDraw.Draw(image,)

    pen_color = random.choice(['#000F55', '#383b3e', '#ac3235'])

    sarabun_font = fonts['sarabun'].get()
    text_faker: Faker = Faker(['th_TH'])

    start_y = 10
    start_x = 60
    max_x = image_width - start_x

    tambon = geo_names.sample()

    text_bboxes = []
    temp_token = {'text': [], 'bbox': BBox()}

    line_hieght = 30
    curr_y = start_y
    curr_x = start_x

    table_flag = False
    table_cells = []
    table_keypoints = []
    for token in text_template_generator.gen():
        font_type = 'sarabun'
        font_color = '#000'
        handwriting_text = ''
        if re.match('^\{[^\}]*\}', token):
            dot_count = 30
            if 'handwriting' in token:
                font_type = 'handwriting'

            if 'tambon' in token:
                token = tambon.TAMBON_T.item().replace('ต.', '')
            elif 'amphoe' in token:
                token = tambon.AMPHOE_T.item().replace('อ.', '')
            elif 'province' in token:
                token = tambon.CHANGWAT_T.item().replace('จ.', '')
            elif 'lastname_th' in token:
                token = random.choice(last_names_th) if random.random(
                ) > .5 else text_faker.last_name()
            elif 'party_name_th' in token:
                token = random.choice(political_parties_th)
            elif 'name_th' in token:
                token = random.choice(names_th) if random.random(
                ) > .5 else text_faker.first_name()
            elif 'title_th' in token:
                token = text_faker.prefix()
            elif 'number' in token:
                if random.random() > .8:
                    number = 0
                # one digit number
                elif random.random() > .5:
                    number = random.randint(1, 9)
                    dot_count = 5
                # two digit number
                elif random.random() > .5:
                    number = random.randint(10, 99)
                    dot_count = 10
                # three digit number
                elif random.random() > .5:
                    number = random.randint(100, 999)
                    dot_count = 10
                else:
                    number = random.randint(0, 9999)
                    dot_count = 10

                if 'reading' in token:
                    number = thai_num2text(number)
                    dot_count = len(number) + random.randint(20, 40)
                elif 'th' in token:
                    number = str(number)
                    number = ''.join([arabic2th(n) for n in number])
                else:
                    number = str(number)
                token = number
            elif 'date' in token:
                number = str(random.randint(1, 31))
                if 'th' in token:
                    number = ''.join([arabic2th(n) for n in number])
                token = number
            elif 'year' in token:
                number = str(random.randint(2500, 2570))
                if 'th' in token:
                    number = ''.join([arabic2th(n) for n in number])
                token = number
            elif 'month_th' in token:
                token = random.choice(months_th)
            elif 'word_n_th' in token:
                token = ''.join(text_faker.words(nb=random.randint(1, 5)))
            elif 'word_th' in token:
                token = text_faker.word()

            if font_type == 'handwriting':
                handwriting_text = token
                token = '.' * dot_count

        # dealing with table
        if token == '%%table' and not table_flag:
            table_flag = True
            continue
        elif token == '%%table' and table_flag:
            table_flag = False
            # filter
            table_cells = [x for x in table_cells if x and x[0]]
            fsize = random.randint(20, 30)
            font = fonts['sarabun'].get(fsize) if random.random(
            ) < 20 else fonts['handwriting'].get(fsize)

            table_width = image_width * 0.92
            table_column_size = [0.15, 0.26, 0.24, 0.35]
            margin = 10

            _start_y = curr_y
            _start_x = (image_width - table_width)/2
            _line_height = 30
            curr_y = _start_y
            for row_i, row in enumerate(table_cells):
                curr_x = _start_x
                row_start = (curr_x, curr_y)
                row_max_y = curr_y + _line_height
                for col_j, (col_size, cell_tokens) in enumerate(zip(table_column_size, row)):
                    cell_width = col_size * table_width
                    lines = []

                    # `text_1` is for handwriting text
                    lines.append({'text': '', 'text_1': None})
                    for token in cell_tokens:
                        # If the token is a separator, add a new line
                        if token['text'] == '<sep>':
                            lines.append({'text': '', 'text_1': None})
                        # Otherwise, append the token's text to the current line
                        else:
                            lines[-1]['text'] += token['text']
                            # If the token has handwriting text, append it to the current line
                            if token['text_1'] is not None:
                                if lines[-1]['text_1'] is None:
                                    lines[-1]['text_1'] = token['text_1']
                                else:
                                    lines[-1]['text_1'] += token['text_1']

                    temp_y = curr_y
                    for line in lines:
                        fsize = random.randint(20, 30)
                        font = fonts['sarabun'].get(fsize) if random.random(
                        ) < .5 else fonts['handwriting'].get(fsize)

                        x0, y0, x1, y1 = draw.textbbox(
                            (curr_x+5, curr_y+5), line['text'], font=font)
                        align_center = random.random() < 0.5
                        if align_center:
                            # center
                            x0 = curr_x + cell_width / 2 - (x1-x0)/2
                        else:
                            # not center
                            x0 = curr_x+5

                        # draw text to image
                        draw.text((x0, curr_y+5,),
                                  line['text'], '#000', font=font)
                        # get text bbox
                        x0, y0, x1, y1 = draw.textbbox(
                            (x0, curr_y+5,), line['text'], font=font)
                        # add text bbox to list
                        text_bboxes.append(
                            {'text': [line['text']], 'bbox': BBox(x0, y0, x1, y1)})

                        # handwriting text
                        if line['text_1']:
                            # get handwriting font
                            hfont = fonts['handwriting'].get(
                                random.randint(20, 34))
                            # get text width
                            txt_w = hfont.getlength(line['text'])
                            xy_pos = (
                                # randomly select an integer between the values of `x0` and `x1-text_width`-10.
                                random.randint(int(x0),
                                               max(int(x0), int(x1-txt_w)-10)),
                                # randomly select an integer between the values of `curr_y-5` and `curr_y+2`.
                                random.randint(curr_y-5, curr_y+2))
                            # draw text with handwriting font
                            draw.text(
                                xy_pos, line['text_1'], '#000', font=hfont)
                            # get text bounding box
                            x0, y0, x1, y1 = draw.textbbox(
                                xy_pos, line['text_1'], font=hfont)
                            # append to text_bboxes
                            text_bboxes.append(
                                {'text': [line['text_1']], 'bbox': BBox(x0, y0, x1, y1)})
                        curr_y += _line_height

                    row_max_y = max(row_max_y, curr_y+10)

                    curr_y = temp_y
                    curr_x += cell_width

                    row_x0, row_y0 = row_start
                    row_x1, row_y1 = curr_x, row_max_y
                    draw.rectangle(
                        (row_x0, row_y0, row_x1, row_y1), outline='#000')
                curr_y = row_max_y
            continue
        elif table_flag:
            if token == '\n':
                table_cells.append([])
                table_cells[-1].append([])
            elif token == '|':
                table_cells[-1].append([])
            else:
                if not table_cells:
                    table_cells.append([])
                if not table_cells[-1]:
                    table_cells[-1].append([])
                table_cells[-1][-1].append({'text': token,
                                           'text_1': handwriting_text})

            continue
        ##

        # Get the bounding box of the text token.
        x0, y0, x1, y1 = draw.textbbox(
            (curr_x, curr_y), token, font=sarabun_font)

        # If the width of the text line is greater than the maximum width,
        # or if the token is a newline character, do the following:
        if x1 > max_x or token == '\n':
            # Get a new font.
            sarabun_font = fonts['sarabun'].get(random.randint(20, 30))

            # Start a new line.
            curr_x = start_x
            curr_y += line_hieght

            # Append the current token to the list of text bboxes.
            text_bboxes.append(temp_token)

            # Create a new empty token dictionary.
            temp_token = {'text': [], 'bbox': BBox()}

            # If the token is a newline character, continue to the next iteration of the loop.
            if token == '\n':
                continue

        # If the token is a tab character, do the following:
        if token == '\t':

            # Tab the current position.
            curr_x += 30

            # Append the current token to the list of text bboxes.
            text_bboxes.append(temp_token)

            # Create a new empty token dictionary.
            temp_token = {'text': [], 'bbox': BBox()}

            # Continue to the next iteration of the loop.
            continue

        # If the token is a space character, do the following:
        elif token == ' ':

            # Advance the current position by the width of a space character.
            curr_x += sarabun_font.getlength(' ')

            # Append the current token to the list of text bboxes.
            text_bboxes.append(temp_token)

            # Create a new empty token dictionary.
            temp_token = {'text': [], 'bbox': BBox()}

            # Get new font.
            sarabun_font = fonts['sarabun'].get(random.randint(17, 24))

            # Continue to the next iteration of the loop.
            continue

        # Get the bounding box of the text token.
        x0, y0, x1, y1 = draw.textbbox(
            (curr_x, curr_y), token, font=sarabun_font)

        # If the font type is "handwriting", do the following:
        if font_type == 'handwriting':
            # Select font
            font = fonts['handwriting'].get(random.randint(20, 27))

            # Append the current token to the list of text bboxes.
            text_bboxes.append(temp_token)

            # Create a new empty token dictionary.
            temp_token = {'text': [], 'bbox': BBox()}

            # Draw a placeholder text, like `....`, for the form text.
            draw.text((curr_x, curr_y), token, font_color, font=sarabun_font)

            # Get the width of the text.
            text_width = font.getlength(handwriting_text)

            # Calculate the position for putting the text.
            xy_pos = (random.randint(int(x0), max(int(x0), int(x1-text_width-10))),
                      random.randint(int(curr_y-10), int(curr_y+2)))

            # Draw the text at the calculated position.
            draw.text(xy_pos, handwriting_text, pen_color, font=font)

            # Get the bounding box of the handwritten text.
            xyxy = draw.textbbox(xy_pos, handwriting_text, font=font)

            # Append the handwritten text to the list of text tokens.
            temp_token['text'].append(handwriting_text)

            # Merge the bounding boxes of the handwritten text and the placeholder.
            temp_token['bbox'].merge(*xyxy)

            # Append the new token dictionary to the list of text bboxes.
            text_bboxes.append(temp_token)

            # Create a new empty token dictionary.
            temp_token = {'text': [], 'bbox': BBox()}

        # Otherwise, do the following:
        else:
            # Draw the text at the current position.
            draw.text((curr_x, curr_y), token, font_color, font=sarabun_font)

            # Append the current token to the list of text tokens.
            temp_token['text'].append(token)

            # Merge the bounding boxes of the text tokens.
            temp_token['bbox'].merge(x0, y0, x1, y1)

        # Update the current position.
        curr_x = x1

        # If current y position is greater than the maximum y position,
        # then stop drawing.
        if curr_y > image.height - start_y:
            break

    if temp_token not in text_bboxes:
        text_bboxes.append(temp_token)

    bboxes_on_image = []
    for x in text_bboxes:
        # if text is empty, then ignore it.
        if not x['text']:
            continue

        joined_text = ''.join(x['text'])

        # If text is `(...)` with any number of `.` in between, then ignore it.
        # Or text is `.` with any number of `.` in between, then ignore it.
        if re.match(r'^\.*\(?\.+\)?\.*$', joined_text) or re.match(r'^\.+$', joined_text):
            continue

        # if text is `(` or `)`, then ignore it.
        if re.match(r'^(\(|\))$', joined_text):
            continue

        bboxes_on_image.append(BoundingBox(
            *x['bbox'].to_list(), label=joined_text))

    np_image = augment_image(np.array(image))
    total_copy = random.randint(0, 3)
    (
        aug_images,
        aug_bboxes,
    ) = seq(
        images=[np_image for _ in range(total_copy)],
        bounding_boxes=[BoundingBoxesOnImage(
            bboxes_on_image, shape=np_image.shape) for _ in range(total_copy)],
    )

    # Save image
    file_saver.save(np_image, BoundingBoxesOnImage(
        bboxes_on_image, shape=np_image.shape))

    # Save augmented images
    for (aug_image, aug_bbox) in zip(aug_images, aug_bboxes):
        file_saver.save(aug_image, aug_bbox
                        .remove_out_of_image_fraction(0.9)
                        .clip_out_of_image())


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

        if segmap is not None:
            cv2.imwrite(os.path.join(self.image_dir,
                                     image_name + '_mask'+'.png'), segmap.get_arr())

        text_instances = [
            {
                'bbox': [bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int],
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, default='vc-dataset')
    argparser.add_argument('--num', type=int, required=True)

    args = argparser.parse_args()
    output_dir = os.path.join('output', args.output_dir)
    file_saver = MMOCRFileSaver(os.path.join(output_dir, 'imgs'),
                                os.path.join(output_dir))

    template_generator = TextTemplate()
    try:
        for x in trange(args.num):
            _main(file_saver, template_generator)
    except KeyboardInterrupt:
        pass

    file_saver.clean(os.path.join(output_dir, 'train.json'))
