"""
作用：遍历整个VOC训练集，生成用于meta tuning的few-shot training set（保存在voclist文件夹中，FSRW原仓库中的data/vocsplit中就是这些文件）
按box：为每个class选择多张图片，最终每个class有k个object
按image：为每个class选择max-k张图片
"""
import argparse
import random
import os
import numpy as np
from os import path

# voc有20个class
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# few_nums = [1, 10] k-shot的k
few_nums = [1, 2, 3, 5, 10]
# few_nums = [20]
DROOT = '/hdd1/hdd_B/bh_data/voc' # 数据集文件夹
root =  DROOT + '/voclist/' # 
rootfile =  DROOT + '/voc_train.txt' # 整个VOC训练集中图片文件路径的voc_train.txt

def is_valid(imgpath, cls_name):
    imgpath = imgpath.strip()
    labpath = imgpath.replace('images', 'labels_1c/{}'.format(cls_name)) \
                         .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
                         .replace('.jpg', '.txt').replace('.png','.txt')
    if os.path.getsize(labpath):
        return True
    else:
        return False

def gen_image_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (images) ------------------')
    for i, clsname in enumerate(classes):
        # i：class的索引
        # clsname：class的名称
        print('===> Processing class: {}'.format(clsname))
        with open(path.join(root, '{}_train.txt'.format(clsname))) as f:
            name_list = f.readlines() # VOC训练集中该class对应所有图片文件的路径
        # max-k
        num = max(few_nums) # 各个k-shot training set中k的最大值
        random.seed(i)
        
        # 为每个class随机选取max-k张图片作为few-shot training set
        # selected_list = random.sample(name_list, num)
        selected_list = []
        while len(selected_list) < num:
            x = random.sample(name_list, num)[0]
            # 如果该图片中没有object（label文件大小为0），则不使用该image，结束本次迭代
            if not is_valid(x, clsname):
                continue
            selected_list.append(x)

        for n in few_nums:
            with open(path.join(root, '{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for i in range(n):
                    f.write(selected_list[i])

# -------------------------------------------------------------------------------------

def get_bbox_fewlist(rootfile, shot):
    """
    IMPORTANT
    # 作用：遍历整个VOC训练集，生成few-shot training set，即为每个class选择多张image，最终每个class有k个object

    # rootfile：整个VOC训练集中图片文件路径的voc_train.txt
    # shot：k-shot的k

    # 结果：返回cls_lists（每个class有1个list，表示few-shot training set中该class对应的所有图片的文件路径）
    """
    with open(rootfile) as f:
        names = f.readlines() # 整个VOC训练集中所有图片文件的路径
    random.seed(2018)

    ### cls_lists：每个class有1个list，表示few-shot training set中该class对应的所有图片的文件路径
    cls_lists = [[] for _ in range(len(classes))]
    cls_counts = [0] * len(classes) # few-shot training set中每个class各有多少个object
    ### 循环至每个class都有k个object
    while min(cls_counts) < shot:
        ### 随机选择1张image并找到其label文件的路径
        imgpath = random.sample(names, 1)[0]
        labpath = imgpath.strip().replace('images', 'labels') \
                                 .replace('JPEGImages', 'labels') \
                                 .replace('.jpg', '.txt').replace('.png','.txt')
        # To avoid duplication 避免重复纳入/检查图片
        names.remove(imgpath) # 本次迭代会判断该图片是否被纳入到few-shot training set中（不管是否纳入该图片，该图片都不用再被检查/纳入了），则将其剔除掉，避免之后重复选择该图片

        ### 如果该图片中没有object（label文件大小为0），则不使用该image，结束本次迭代
        if not os.path.getsize(labpath):
            continue
        
        ### 如果该图片中object多于3个，则不使用该image，结束本次迭代
        # Load converted annotations
        bs = np.loadtxt(labpath) # 加载该图片的label
        bs = np.reshape(bs, (-1, 5))
        if bs.shape[0] > 3:
            continue

        ### 如果将该图片包括到few-shot training set后，会超过k-shot的k，则不使用该image，结束本次迭代
        # Check total number of bbox per class so far
        overflow = False
        bcls = bs[:,0].astype(np.int).tolist() # 该image中各个object的class label
        for ci in set(bcls):
            if cls_counts[ci] + bcls.count(ci) > shot: # 判断将该图片包括到few-shot train set后，会不会超过k-shot的k
                overflow = True
                break
        if overflow:
            continue

        # 将该图片纳入到其中object的class对应的image list中，即该图片作为其中object的class的训练图片之一
        # Add current imagepath to the file lists 
        for ci in set(bcls):
            cls_counts[ci] += bcls.count(ci)
            cls_lists[ci].append(imgpath)

    ### cls_lists：每个class有1个list，表示few-shot training set中该class对应的所有图片的文件路径
    return cls_lists


def gen_bbox_fewlist():
    """
    # 作用：遍历整个VOC训练集，生成few-shot training set，即为每个class选择多张image，最终每个class有k个object
    # 结果：将生成的few-shot training set保存在voclist中，格式如'box_{k}shot_{clsname}_train.txt'
    """
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    for n in few_nums:
        # n：k-shot的k
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlist(rootfile, n) # 每个class有1个list，表示few-shot training set中该class对应的所有图片的文件路径
        for i, clsname in enumerate(classes):
            # i：class的索引
            # clsname：class的名称
            print('   | Processing class: {}'.format(clsname))
            with open(path.join(root, 'box_{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    # name：few-shot training set中该class对应的某图片的文件路径
                    f.write(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None, choices=['box', 'img', 'both'])
    args = parser.parse_args()

    # 默认按box生成k-shot training set
    if args.type is None or args.type == 'box':
        gen_bbox_fewlist()
    # 按image生成k-shot training set
    elif args.type == 'img':
        gen_image_fewlist()
    elif args.type == 'both':
        gen_image_fewlist()
        gen_bbox_fewlist()


if __name__ == '__main__':
    main()
