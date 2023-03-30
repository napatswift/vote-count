from PIL import Image, ImageFont, ImageDraw
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug import augmenters as iaa
import random
import attacut
import re
import pandas as pd
import os
from tqdm import trange
import numpy as np
import cv2

last_names_th = open('family_names_th.txt').read().split('\n')[:-1]
names_th = open('female_names_th.txt').read().split('\n')[:-1]
geo_names = pd.read_csv('tambons.csv')
months_th = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
             'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
political_parties_th = ['ประชาธิปัตย์', 'ประชากรไทย', 'ความหวังใหม่', 'เครือข่ายชาวนาแห่งประเทศไทย', 'เพื่อไทย', 'เพื่อแผ่นดิน', 'ชาติพัฒนา', 'ชาติไทยพัฒนา', 'อนาคตไทย', 'ภูมิใจไทย', 'สังคมประชาธิปไตยไทย', 'ประชาสามัคคี', 'ประชาธิปไตยใหม่', 'พลังชล', 'ครูไทยเพื่อประชาชน', 'พลังสหกรณ์', 'พลังท้องถิ่นไท', 'ถิ่นกาขาวชาววิไล',
                        'รักษ์ผืนป่าประเทศไทย', 'ไทรักธรรม', 'เสรีรวมไทย', 'รักษ์ธรรม', 'เพื่อชาติ', 'พลังประชาธิปไตย', 'ภราดรภาพ', 'พลังไทยรักชาติ', 'เพื่อชีวิตใหม่', 'ก้าวไกล', 'ทางเลือกใหม่', 'ประชาภิวัฒน์', 'พลเมืองไทย', 'พลังไทยนำไทย', 'พลังธรรมใหม่', 'ไทยธรรม', 'ไทยศรีวิไลย์', 'รวมพลังประชาชาติไทย', 'สยามพัฒนา', 'เพื่อคนไทย', 'พลังปวงชนไทย', 'พลังไทยรักไทย']

seq = iaa.Sequential([
    iaa.PiecewiseAffine(scale=(0.01, 0.03)),
    iaa.Resize((0.2, 1.0)),
    iaa.ElasticTransformation(alpha=200, sigma=40),
    iaa.pillike.EnhanceBrightness(factor=(0.6, 1.5)),
    iaa.CoarseDropout((0.0005, 0.001), size_percent=0.4),
    iaa.BlendAlphaMask(
        iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
        iaa.GaussianBlur(sigma=(1, 1.6)),
    ),
    iaa.pillike.FilterSharpen(5),
    iaa.Affine(rotate=(-2, 2)),
    iaa.ElasticTransformation(alpha=200, sigma=90),
    iaa.AverageBlur(k=(2, 11)),
    iaa.BlendAlphaMask(
          iaa.InvertMaskGen(0.8, iaa.VerticalLinearGradientMaskGen()),
          iaa.MotionBlur(k=(3,5))
    ),
])

class BBox:
    def __init__(self, x0=None, y0=None, x1=None, y1=None):
        self.x0 = x0
        self.x1 = y0
        self.y0 = x1
        self.y1 = y1
        pass

    def merge(self, x0, y0, x1, y1):
        if self.x0 is None:
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
        self.font_paths = [os.path.join(font_dir, fp)for fp in os.listdir(font_dir) if fp.endswith('.ttf')]

    def get(self, size=24):
        return ImageFont.truetype(random.choice(self.font_paths), size=size)

def arabic2th(n):
    return chr(ord(n)+(ord('๑')-ord('1')))

# https://suilad.wordpress.com/2012/05/09/code-python-%E0%B9%81%E0%B8%9B%E0%B8%A5%E0%B8%87%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B9%80%E0%B8%A5%E0%B8%82-%E0%B9%84%E0%B8%9B%E0%B9%80%E0%B8%9B%E0%B9%87%E0%B8%99-%E0%B8%95%E0%B8%B1%E0%B8%A7%E0%B8%AB/
thai_number = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
unit = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน")

def unit_process(val):
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
    n_list = [s_number[i:i + 6].rstrip("0") for i in range(0, len(s_number), 6)]
    result = unit_process(n_list.pop(0))

    for i in n_list:
        result = unit_process(i) + 'ล้าน' + result

    return result

fonts = {'sarabun':RandomFont('formal'), 'handwriting': RandomFont('handwriting'),}

image_width = 826
image_height = 1169

text_templates = ["""


\t\t\t\t\tรายงานผลการนับคะแนนสมาชิกสภาผู้แทนราษฎรแบบแบ่งเขตเลือกตั้ง
\t\t\t\t\t\t\t\t\t-----------------------------------
\tตามที่ได้มีพระราชกฤษฎีกาให้มีการเลือกตั้งสมาชิกสภาผู้แทนราษฎรและคณะกรรมการการเลือกตั้งได้กำหนดให้วันที่ {date_th} เดือน {month_th} พ.ศ. {year_th} เป็นวันเลือกตั้ง
\tบัดนี้ คณะกรรมการประจำหน่วยเลือกตั้งใด้ดำเนินการนับคะแบนสมาชิกสภาผู้แทนราษฎรแบบแบ่งเขตเลือกตั้งของหน่วยเลือกตั้งที่ {number_handwriting} หมู่ที่ {number_handwriting} ตำบล/เทศบาล {tambon_name_handwriting_th} อำเภอ {amphoe_handwriting_th} เขตเลือกตั้งที่ {number_th} จังหวัด {province_th} เสร็จสิ้นเป็นที่เรียบร้อยแล้ว ดังนั้น จึงขอรายงานผลการนับคะแนนของหน่วยเลือกตั้งดังกล่าว ดังนี้
\t๑. จำนวนผู้มีสิทธิเลือกตั้ง
\t\t๑.๑ จำนวนผู้มีสิทธิเลือกตั้งตามบัญชีรายชื่อผู้มิสิทธิเลือกตั้ง {number_handwriting} คน
({number_reading_handwriting_th})
\t\t๑.๒ จำนวนผู้มีสิทธิเลือกตั้งที่มาแสดงตน {number_handwriting} คน ({number_reading_handwriting_th}) (เฉพาะวันเลือกตั้ง)
\t๒. จำนวนบัตรเลือกตั้ง
\t\t๒.๑ จำนวนบัตรเลือกตั้งที่ได้รับจัดสรร {number_handwriting} บัตร ({number_reading_handwriting_th})
\t\t๒.๒ จำนวนบัตรเลือกตั้งที่ใช้ {number_handwriting} บัตร ({number_reading_handwriting_th})
\t\t\t๒.๒.๑ บัตรดี {number_handwriting} บัตร ({number_reading_handwriting_th})
\t\t\t๒.๒.๒ บัตรเสีย {number_handwriting} บัตร ({number_reading_handwriting_th})
\t\t\t๒.๒.๓ บัตรไม่เลือกผู้สมัครใด {number_handwriting} บัตร ({number_reading_handwriting_th})
\t\t๒.๓ จำนวนบัตรเลือกตั้งที่เหลือ {number_handwriting} บัตร ({number_reading_handwriting_th})
\t๓. จำนวนคะแนนที่ผู้สมัครรับเลือกตั้งแต่ละคนได้รับเรียงตามลำดับหมายเลขประจำตัวผู้สมัคร

%%table
หมายเลข<sep>ประจำตัวผู้สมัคร|ชื่อ-สกุล<sep>ผู้สมัครรับเลือกตั้ง|สังกัด<sep>พรรคการเมือง|ได้คะแนน<sep>(ให้กรอกทั้งตัวเลขและตัวอักษร)
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
%%table
ชุดที่ ๒ ปิดประกาศ ณ ที่เลือกตั้ง
""",
"""
%%table
หมายเลข<sep>ประจำตัวผู้สมัคร|ชื่อ-สกุล<sep>ผู้สมัครรับเลือกตั้ง|สังกัด<sep>พรรคการเมือง|ได้คะแนน<sep>(ให้กรอกทั้งตัวเลขและตัวอักษร)
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
%%table
ชุดที่ ๒ ปิดประกาศ ณ ที่เลือกตั้ง
""",
"""
%%table
หมายเลข<sep>ประจำตัวผู้สมัคร|ชื่อ-สกุล<sep>ผู้สมัครรับเลือกตั้ง|สังกัด<sep>พรรคการเมือง|ได้คะแนน<sep>(ให้กรอกทั้งตัวเลขและตัวอักษร)
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|({number_reading_number_handwriting_th})
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
{number_th}|{title_th}{name_th} {lastname_th}|{party_name_th}|{number_handwriting}
%%table
ชุดที่ ๒ ปิดประกาศ ณ ที่เลือกตั้ง
"""
]

def get_tokens():
    text_template = random.choice(text_templates)
    big_pieces = re.split('(\<[^\>]*\>|\t|\n| |\{[^\}]*\}|%%\w+|\|)', text_template)
    tokens = []

    for piece in big_pieces:
        if re.findall('[\u0E00-\u0E0F]+', piece):
            tokens.extend(attacut.tokenize(piece))
        elif piece == '':
            continue
        else:
            tokens.append(piece)
    return tokens

def _(image_id):#create blank white paper
    image = Image.new('RGB', (image_width, image_height,), random.choice(['#00FF7F', 'white']))
    table_mask = Image.new('L', (image_width, image_height,), '#000')
    draw = ImageDraw.Draw(image,)
    table_draw = ImageDraw.Draw(table_mask,)

    pen_color = random.choice(['#000F55', '#383b3e', '#ac3235'])
    sarabun_font = fonts['sarabun'].get()

    start_y = 10
    start_x = 60
    max_x = image_width - start_x

    tambon = geo_names.sample()

    text_bboxes = []
    temp_token = {'text': [],'bbox': BBox()}

    line_hieght = 30
    curr_y = start_y
    curr_x = start_x

    table_flag = False
    table_cells = []
    table_keypoints = []
    for token in get_tokens():
        font_type = 'sarabun'
        font_color = '#000'
        handwriting_text = ''
        if re.match('^\{[^\}]*\}',token):
            dot_count = 30
            if 'handwriting' in token:
                font_type = 'handwriting'
            if 'tambon' in token:
                token = tambon.TAMBON_T.item().replace('ต.','')
            elif 'amphoe' in token:
                token = tambon.AMPHOE_T.item().replace('อ.','')
            elif 'province' in token:
                token = tambon.CHANGWAT_T.item().replace('จ.','')
            elif 'lastname_th' in token:
                token = random.choice(last_names_th)
            elif 'party_name_th' in token:
                token = random.choice(political_parties_th)
            elif 'name_th' in token:
                token = random.choice(names_th)
            elif 'title_th' in token:
                token = random.choice(['นาย','นางสาว','นาง'])
            elif 'number' in token:
                number = random.randint(1,100)
                dot_count = 10
                if 'reading' in token:
                    number = thai_num2text(number)
                    dot_count = 50
                elif 'th' in token:
                    number = str(number)
                    number = ''.join([arabic2th(n) for n in number])
                else:
                    number = str(number)
                token = number
            elif 'date' in token:
                number = str(random.randint(1,31))
                if 'th' in token:
                    number = ''.join([arabic2th(n) for n in number])
                token = number
            elif 'year' in token:
                number = str(random.randint(2500,2570))
                if 'th' in token:
                    number = ''.join([arabic2th(n) for n in number])
                token = number
            elif 'month_th' in token:
                token = random.choice(months_th)

            if font_type == 'handwriting':
                handwriting_text = token            
                token = '.' * dot_count

        ## dealing with table
        if token == '%%table' and not table_flag:
            table_flag = True
            continue
        elif token == '%%table' and table_flag:
            table_flag = False
            # filter
            table_cells = [x for x in table_cells if x and x[0]]
            font=fonts['sarabun'].get(22)

            table_width = image_width * 0.92
            table_column_size = [0.15,0.26,0.24,0.35]
            margin = 10

            _start_y = curr_y
            _start_x = (image_width - table_width)/2
            _line_height = 30
            curr_y  = _start_y
            for row_i, row in enumerate(table_cells):
                curr_x = _start_x
                row_start = (curr_x, curr_y)
                row_max_y = curr_y + _line_height
                for col_j, (col_size, cell_tokens) in enumerate(zip(table_column_size, row)):
                    cell_width = col_size * table_width
                    lines = []
                    lines.append({'text': '', 'text_1': None})
                    for token in cell_tokens:
                        if token['text'] == '<sep>':
                            lines.append({'text': '', 'text_1': None})
                        else:
                            lines[-1]['text'] += token['text']
                            if token['text_1'] is None:
                                continue
                            if lines[-1]['text_1'] is None :
                                lines[-1]['text_1'] = token['text_1']
                            else:
                                lines[-1]['text_1'] += token['text_1']

                    temp_y = curr_y
                    for line in lines:
                        x0,y0,x1,y1 = draw.textbbox((curr_x+5, curr_y+5), line['text'], font=font)
                        align_center = True
                        if align_center:
                            # center
                            x0 = curr_x + cell_width / 2 - (x1-x0)/2
                        else:
                            # not center
                            x0 = curr_x+5
                        draw.text((x0, curr_y+5,), line['text'], '#000', font=font)
                        x0,y0,x1,y1 = draw.textbbox((x0, curr_y+5,), line['text'], font=font)
                        bbox = BBox(x0,y0,x1,y1)
                        text_bboxes.append({'text': [line['text']], 'bbox': BBox(x0,y0,x1,y1)})
                        if line['text_1']:
                            hfont = fonts['handwriting'].get()
                            text_width = hfont.getlength(line['text_1'])
                            xy_pos = (random.randint(int(x0), max(int(x0), int(x1-text_width)-10)),
                                    random.randint(curr_y-5, curr_y+2))
                            draw.text(xy_pos, line['text_1'], '#000', font=hfont)
                            x0,y0,x1,y1 = draw.textbbox(xy_pos, line['text_1'], font=hfont)
                            text_bboxes.append({'text': [line['text_1']], 'bbox': BBox(x0,y0,x1,y1)})
                        curr_y += _line_height

                    row_max_y = max(row_max_y, curr_y+10)

                    curr_y = temp_y
                    curr_x += cell_width
                    draw.rectangle((*row_start, curr_x, row_max_y), outline='#000')
                    table_draw.rectangle((*row_start, curr_x, row_max_y), outline='#fff')
                    if row_i == 0:
                        row_x0, row_y0 = row_start
                        if col_j == 0:
                            table_keypoints.append((row_x0, row_y0))
                        table_keypoints.append((curr_x, row_y0))
                    elif row_i == len(table_cells) - 1:
                        if col_j == 0:
                            row_x0, row_y0 = row_start
                            table_keypoints.append((row_x0, row_max_y))
                        table_keypoints.append((curr_x, row_max_y))
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
                table_cells[-1][-1].append({'text':token, 'text_1': handwriting_text})

            continue
        ##
        
        x0,y0,x1,y1 = draw.textbbox((curr_x,curr_y), token, font=fonts['sarabun'].get())
        
        if x1 > max_x or token == '\n':
            # start new line
            curr_x = start_x 
            curr_y += line_hieght
            text_bboxes.append(temp_token)
            temp_token = {'text': [],'bbox': BBox()}
            if token == '\n': continue
        if token == '\t':
            # tab
            text_bboxes.append(temp_token)
            temp_token = {'text': [],'bbox': BBox()}
            curr_x += 30; continue

        elif token == ' ':
            text_bboxes.append(temp_token)
            temp_token = {'text': [],'bbox': BBox()}
            curr_x += fonts['sarabun'].get().getlength(' ')
            
            continue
        
        x0,y0,x1,y1 = draw.textbbox((curr_x,curr_y), token, font=sarabun_font)
        if font_type == 'handwriting':
            font=fonts['handwriting'].get()
            text_bboxes.append(temp_token)
            temp_token = {'text': [],'bbox': BBox()}
            # draw ............... for forms placeholder
            draw.text((curr_x, curr_y), token, font_color, font=sarabun_font)
            # get the text width
            text_width = font.getlength(handwriting_text)
            # calculate position for putting text
            xy_pos = (random.randint(int(x0), max(int(x0), int(x1-text_width-10))), random.randint(int(curr_y-10), int(curr_y+2)))
            draw.text(xy_pos,handwriting_text, pen_color, font=font)
            xyxy = draw.textbbox(xy_pos, handwriting_text, font=font)
            temp_token['text'].append(handwriting_text)
            temp_token['bbox'].merge(*xyxy)
            text_bboxes.append(temp_token)
            temp_token = {'text': [],'bbox': BBox()}
        else:
            draw.text((curr_x, curr_y), token, font_color, font=sarabun_font)
            temp_token['text'].append(token)
            temp_token['bbox'].merge(x0,y0,x1,y1)

        curr_x = x1

    bboxes_on_image = []
    for x in text_bboxes:
        if x['text']:
            bboxes_on_image.append(BoundingBox(*x['bbox'].to_list(), ''.join(x['text'])))

    split = 'training' if random.random() > .2 else 'test'

    image_dir = f'output/dataset/withmask_{split}_images'
    os.makedirs(image_dir, exist_ok=True)

    localization_dir = f'output/dataset/ch4_{split}_localization_transcription_gt'
    os.makedirs(localization_dir, exist_ok=True)

    keypoints = [Keypoint(x,y) for x,y in set(table_keypoints)]
    np_image = np.array(image)
    aug_images, aug_bboxes, aug_segmaps = seq(
        images=[np_image for _ in range(5)],
        bounding_boxes=[BoundingBoxesOnImage(bboxes_on_image, shape=np_image.shape) for _ in range(5)],
        segmentation_maps=[SegmentationMapsOnImage(np.array(table_mask)[:,:,None], shape=np_image.shape) for _ in range(5)],
        keypoints=[KeypointsOnImage(keypoints, np_image.shape) for _ in range(5)]
    )

    for i, (aug_image, aug_bbox, aug_segmap) in enumerate(zip(aug_images, aug_bboxes, aug_segmaps)):
        image_name = f'img_{image_id}{i}'
        cv2.imwrite(os.path.join(image_dir, image_name+'.jpg'), aug_image)
        cv2.imwrite(os.path.join(image_dir, image_name+'_mask'+'.jpg'), aug_segmap.get_arr())
        
        fp = open(os.path.join(localization_dir, 'gt_'+image_name+'.txt'), 'w')
        for bbox in aug_bbox.bounding_boxes:
            for point in bbox.to_keypoints():
                fp.write(str(point.x_int)+',')
                fp.write(str(point.y_int)+',')

            fp.write(bbox.label)
            fp.write('\n')
        fp.close()

for x in trange(1):
    _(x)