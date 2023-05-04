import json
import argparse
import os
import shutil

def read_dataset(json_path: str):
    if json_path is None:
       return None
    
    with open(json_path, 'r') as fp:  
      data = json.load(fp)
    return data

def get_dataset_file(dir_path):
  files = os.listdir(dir_path)
  return [os.path.join(dir_path, f) for f in files if f.endswith('.json')]

def concat_dataset(json_1: str, json_2, dest):
  ds1 = read_dataset(json_1)
  ds2 = read_dataset(json_2)

  data_list = []
  metadata = ds1['metainfo']

  dl1 = ds1['data_list']
  print('dataset 1: size of {}'.format(len(dl1)))
  move_image_file(os.path.dirname(json_1), dest, dl1)
  data_list += dl1

  if ds2 is not None:  
    dl2 = ds2['data_list']
    print('dataset 2: size of {}'.format(len(dl2)))
    move_image_file(os.path.dirname(json_2), dest, dl2)
    data_list += dl2
  
  print('concat dataset: size of {}'.format(len(data_list)))

  return dict(metainfo=metadata, data_list=data_list)

def move_image_file(source, dest, data_list):
  out_dir = os.path.join(dest, 'imgs', source)
  os.makedirs(out_dir, exist_ok=True)
  fnames = []

  for data in data_list:
    image_path = data['img_path']
    
    if image_path in fnames:
       del data
       continue
    fnames.append(image_path)

    dest_name = os.path.join(out_dir, image_path)

    if os.path.isfile(dest_name):
      raise FileExistsError(
          'destination ({}) file already exists, try change the dataset path'.format(dest_name)
        )

    shutil.copy2(f'{source}/imgs/{image_path}', dest_name)

    data['img_path'] = os.path.join(source, image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_1', help='path to dataset directory')
    parser.add_argument('ds_2', help='path to dataset directory')
    parser.add_argument('dest', help='path to dataset directory')
    args = parser.parse_args()

    dataset_list = dict()
    
    ds_files = get_dataset_file(args.ds_1) + get_dataset_file(args.ds_2)

    for f in ds_files:
        if 'train' in f:
          split = 'train'
        elif 'test' in f:
           split = 'test'
        if split not in dataset_list.keys():
           dataset_list[split] = []
        split_list = dataset_list.get(split)
        split_list.append(f)

    os.makedirs(args.dest,)

    for split in dataset_list:
      print(f'concat {split} dataset')
      if len(dataset_list[split]) == 1:
         dataset_list[split].append(None)

      concat_ds = concat_dataset(*dataset_list[split], args.dest)

      with open(os.path.join(args.dest, f'textdet_{split}.json'), 'w') as fp:
         json.dump(concat_ds, fp, ensure_ascii=False)
        
