"""
需要修改的地方：
1. sets中替换为自己的数据集
3. 将本文件放到VOC2007目录下
4. 直接开始运行
"""

import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir
from os.path import join
import random

is_create_val = 1 #是否进行划分
sets = ['val','train']  #替换为自己的数据集
path = ''
random.seed(2024)
xml_dir = '/home/chen/yolo/ultralytics/data/A_11_image'  #  标签文件地址
img_dir = '/home/chen/yolo/ultralytics/data/A_11_xml'  # 图像文件地址
txt_dir = '/home/chen/yolo/ultralytics/data/A_11_label' #输出yolo格式标签文件的地址

def convert(size, box_x, box_y):
    # 进行归一化,并算出detection框的大小
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x_convert = [i * dw for i in box_x]
    y_convert = [i * dh for i in box_y]

    roi_w = (max(box_x) - min(box_x)) * dw + 0.02
    roi_h = (max(box_y) - min(box_y)) * dh + 0.02 #0.02为补偿，防止切割目标图像外围
    roi_center_x = (max(box_x) + min(box_x)) / 2 * dw
    roi_center_y = (max(box_y) + min(box_y)) / 2 * dh

    final = [roi_center_x, roi_center_y, roi_w, roi_h]
    for i in range(0, len(x_convert)):
        final.append(float(x_convert[i]))
        final.append(float(y_convert[i]))
        final.append(2.000000)

    return final


def convert_annotation(img_name, classes):
    in_file = open('%s/%s.xml' %(xml_dir ,img_name), 'r', encoding='utf-8')
    out_file = open('%s/%s.txt' %(txt_dir , img_name), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        #     continue
        if int(difficult) == 1:
            continue
        # cls_id = classes.index(cls)
        # ！！！！注意：这里把类别写死了，可能要改！！！！
        cls_id = 0
        xmlbox = obj.find('bndbox')
        box_x = [float(xmlbox.find('x1').text),float(xmlbox.find('x2').text),
                float(xmlbox.find('x3').text),float(xmlbox.find('x4').text)]
        box_y = [float(xmlbox.find('y1').text),float(xmlbox.find('y2').text),
                float(xmlbox.find('y3').text),float(xmlbox.find('y4').text)]

        bb = convert((w, h), box_x, box_y)
        out_file.write(
            str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')

# # 打开txt文件
# # classes:red_target
# def gen_voc_lable(classes):
#     for image_set in sets:
#         if not os.path.exists('voc_to_yolo/'):
#             os.makedirs('voc_to_yolo/')
#         # image_paths = open('%s.txt' %
#         #                  (image_set)).readlines()  #.strip()#.split()
#         # print(image_ids)
#         xml_paths = open('%s.txt' %
#                          (image_set)).readlines()  #.strip()#.split()
#         print('*' * 20)
#         for xml_path in xml_paths:
#             xml_path = xml_path[:-1]
#             print(xml_path)
#             convert_annotation(xml_path, classes)

# xml_dir = '/home/hiling/datasets/engine-test/labels/train'  #  标签文件地址
# img_dir = '/home/hiling/datasets/engine-test/images/train'  # 图像文件地址
# txt_dir = '/home/hiling/datasets/engine-test/labels/train/to_yolo' #输出yolo格式标签文件的地址
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)
path_list = list()
# 依次对文件进行操作
for xml in os.listdir(xml_dir):
    # xml_path = os.path.join(xml_dir, xml)
    img_name = xml.split('.')[0]
    convert_annotation(img_name, 'asd')

# random.shuffle(path_list)
# ratio = 0.9

# for xml in path_list:
