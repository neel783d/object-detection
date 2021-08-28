import argparse
import os
from xml.etree import ElementTree as ET

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2,
               'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
               'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13,
               'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17,
               'train': 18, 'tvmonitor': 19}


def parse_data(basedir, image_id):
  # Processes Each XML files and map it to it's JPEG Path
  xml_fp = f'{basedir}/Annotations/{image_id}.xml'
  jpeg_fp = f'{basedir}/JPEGImages/{image_id}.jpg'

  tree = ET.parse(xml_fp)
  root = tree.getroot()

  data = []
  for obj in root.iter('object'):
    cls = obj.find('name').text
    id = classes_num.get(cls, -1)
    if id < 0:
      continue

    box = obj.find('bndbox')
    bindex = list(map(lambda x: int(box.find(x).text),
                      ['xmin', 'xmax', 'ymin', 'ymax']))
    bindex = list(map(str, bindex))
    out_str = '\t'.join([cls, str(id), jpeg_fp] + bindex)
    data.append(out_str)
  return data


def get_ids_by_set(basedir):
  dir = f'{basedir}/ImageSets/Main'
  ids = {'val': [], 'train': [], 'test': []}
  for id in ids.keys():
    image_ids = list(
        a.strip().split(' ')[0] for a in
        open(f'{dir}/{id}.txt', 'r').readlines() if
        a.strip().split(' ')[0] != '')

    ids[id] = image_ids
  return ids


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-basedir', default='VOCdevkit/VOC2007')
  parser.add_argument('-outdir', default='data')
  args = parser.parse_args()
  return args


def generate_processed_tsv():
  args = parse_arguments()
  ids = get_ids_by_set(args.basedir)

  if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

  processed = {'val': [], 'train': [], 'test': []}
  count = 0
  for set_id, image_ids in ids.items():
    old_data = processed[set_id]
    for image_id in image_ids:
      data = parse_data(args.basedir, image_id)
      old_data += data

      if count % 100 == 0:
        print(f'processed data: {count}')
      count += 1

  for set_id, image_ids in processed.items():
    with open(f'{args.outdir}/{set_id}.tsv', 'w') as out:
      out.write('\n'.join(image_ids))
      out.flush()

  print('-' * 50)
  print('\n'.join(f'{set_id}: {len(processed[set_id])}'
                  for set_id in processed.keys()))


if __name__ == '__main__':
  generate_processed_tsv()
