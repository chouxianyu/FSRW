"""
该文件应该在$DATA_ROOT（即数据集根文件夹）中运行
作用：
在当前文件夹生成名如2007_train.txt的文件（内容为对应年份和对应set中图片的路径）
将XML格式的annotation转换为txt格式的label，在VOC2007和VOC2012中生成labels文件夹（里面有多个.txt文件，保存所有图片的label）
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

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

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# 获取当前文件夹路径（数据集根路径，VOC）
wd = getcwd()

for year, image_set in sets:
    # year：年份（2007或2012）
    # image_set：set（train、val或test）

    # 为VOC2007和VOC2012分别创建1个文件夹labels
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    # 读取该年份该set中所有图片的id（和annotation和label文件的名称对应）
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    
    
    list_file = open('%s_%s.txt'%(year, image_set), 'w') # 生成该年份该set中所有图片的路径
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id) # 将该年份该set中该图片的annotation（.xml）转为label（.txt）
    list_file.close()

