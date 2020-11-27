import xml.etree.ElementTree as ET
import os, re
from pathlib import Path


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_labels = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text
        label = boxes.find("name").text

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_labels.append(label)

    return filename, list_with_all_boxes, list_with_all_labels


def get_project_root():
    return Path(__file__).parent.parent


def read_contents(xml_dir):
    if not os.path.isdir(xml_dir):
        print('{} is not a directory'.format(xml_dir))
    else:
        xml_files = [f for f in os.listdir(xml_dir) if re.match(r'.+.xml$', f)]
    parsed_imgs = []
    for xml_file in xml_files:
        xml_file = os.path.join(xml_dir, xml_file)
        parsed_imgs.append(read_content(xml_file))

    return parsed_imgs


if __name__ == '__main__':
    xml_dir = os.path.join(get_project_root(), 'data/annotations')
    parsed_imgs = read_contents(xml_dir)
    print(parsed_imgs)
