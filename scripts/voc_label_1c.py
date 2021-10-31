"""
该文件应该在$DATA_ROOT（即数据集根文件夹）中运行
作用：
生成few-shot training
在VOC2007和VOC2012中生成labels_1c文件夹，里面有20个文件夹，每个文件夹对应1个class（文件夹命名为class名称，里面有很多个txt，txt命名为图片id，txt内容为该图片中属于该class的object的label），用于meta input
创建文件夹voclist，里面是很多个.txt：一种是各年各set各class的图片文件的路径，一种是各set各class（2年合起来）的图片文件的路径
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--type', type=str, choices=['1c', 'all'], required=True)
# args = parser.parse_args()

# 数据集：VOC分为2007和2012
# 2007包括train、val和test
# 2012只包括train和val
# 一般使用2007和2012的train和val作为训练集，使用2007的test作为测试集
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# VOC数据集共包括20个class
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    # box center的坐标
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw # 使用图片的width进行normalization
    w = w*dw # 使用图片的width进行normalization
    y = y*dh # 使用图片的height进行normalization
    h = h*dh # 使用图片的height进行normalization
    return (x,y,w,h)

def convert_annotation(year, image_id, class_name):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels_1c/%s/%s.txt'%(year, class_name, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls != class_name or int(difficult) == 1:
            continue
        # cls_id = classes.index(cls)
        cls_id = 0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 获取当前文件夹路径（数据集根路径，VOC）
wd = getcwd()

# 创建文件夹voclist
if not os.path.exists('voclist'):
    os.mkdir('voclist')

for class_name in classes:
    for year, image_set in sets:
        # class_name：class名称（共有20个class）
        # year：年份（2007或2012）
        # image_set：set（train、val或test）
        
        
        # 读取该年份该set中该class所有图片的id（和annotation和label文件的名称对应）
        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year, class_name, image_set)).read().strip().split()
        ids, flags = image_ids[::2], image_ids[1::2] # 每个txt中，每行有2个元素、格式为(image_id, flag)
        image_ids = list(zip(ids, flags))

        # File to save the image path list
        # 该txt保存该年该set该class的图片文件的路径
        list_file = open('voclist/%s_%s_%s.txt'%(year, class_name, image_set), 'w')

        # File to save the image labels，创建labels_1c中的20个文件夹
        label_dir = 'labels_1c/' + class_name
        if not os.path.exists('VOCdevkit/VOC%s/%s/'%(year, label_dir)):
            os.makedirs('VOCdevkit/VOC%s/%s/'%(year, label_dir))

        # Traverse all images
        for image_id, flag in image_ids:
            if int(flag) == -1:
                continue
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
            convert_annotation(year, image_id, class_name)
        list_file.close()

    # 各set各class（2年合起来）的图片文件的路径
    files = [
        'voclist/2007_{}_train.txt'.format(class_name),
        'voclist/2007_{}_val.txt'.format(class_name),
        'voclist/2012_{}_*.txt'.format(class_name)
    ]
    files = ' '.join(files)
    cmd = 'cat ' + files + '> voclist/{}_train.txt'.format(class_name)
    os.system(cmd)