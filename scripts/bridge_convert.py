import xml.etree.ElementTree as ET
import os, re
from pathlib import Path
import tensorflow as tf
import numpy as np

CLASSES = ['SedanCar',
           'SUV',
           'MiniBus',
           '2A_4WTruck',
           '2A_6WTruck',
           '3ATruck',
           '4ATruck',
           '5ATruck',
           '6ATruck',
           'Others']

NUM_CLASSES = len(CLASSES)
TRAIN_INPUT_SIZE = 416
batch_size = 2
STRIDES = [8, 16, 32]
ANCHOR_PER_SCALE = 3




def label2id():
    label_to_id = {}
    for id, label in enumerate(CLASSES):
        label_to_id[label] = id
    return label_to_id


def id2label():
    id_to_label = {}
    for id, label in enumerate(CLASSES):
        id_to_label[id] = label
    return id_to_label


def read_content(xml_file, img_dir, count):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    if re.match(r'^IP.+', filename):
        return None
    if re.match(r'.+\.jpeg$', filename):
        filename = filename[:-4] + 'jpg'
    img_path = os.path.join(get_project_root(), img_dir, filename)
    if not os.path.exists(img_path):
        print(filename)
    txt = img_path
    label_to_id = label2id()
    for boxes in root.iter('object'):
        ymin = boxes.find("bndbox/ymin").text
        xmin = boxes.find("bndbox/xmin").text
        ymax = boxes.find("bndbox/ymax").text
        xmax = boxes.find("bndbox/xmax").text
        label = boxes.find("name").text
        if re.match(r'.+_in$', label):
            label = label[:-3]
        elif re.match(r'.+_out$', label):
            label = label[:-4]
        if re.match('others', label):
            label = 'Others'
        id = label_to_id[label]
        txt += ' ' + ','.join([xmin, ymin, xmax, ymax, str(id)])
    return txt  + '\n'


def get_project_root():
    return Path(__file__).parent.parent


def convert_contents(xml_dir, img_dir, dataset_path, num_pics_to_train):
    if not os.path.isdir(xml_dir):
        print('{} is not a directory'.format(xml_dir))
    else:
        xml_files = [f for f in os.listdir(xml_dir) if re.match(r'.+.xml$', f)]

    print('total_pics: {}'.format(len(xml_files)))
    buffer = ''
    xml_files.sort()
    l = min(len(xml_files), num_pics_to_train)
    sel_xml_files = xml_files[:l]
    for id, xml_file in enumerate(sel_xml_files):
        xml_file = os.path.join(xml_dir, xml_file)
        txt = read_content(xml_file, img_dir, id)
        if txt is not None:
            buffer += txt
    with open(dataset_path, 'w') as dataset_txt:
        dataset_txt.write(buffer)


def convert_annotation_to_dataset(num_pics_to_train):
    xml_dir = os.path.join(get_project_root(), 'data/tag')
    dataset_path = os.path.join(get_project_root(), 'data/dataset/bridge.txt')
    img_dir = os.path.join(get_project_root(), 'data/images')
    convert_contents(xml_dir, img_dir, dataset_path, num_pics_to_train)


if __name__ == '__main__':
    convert_annotation_to_dataset(500)

