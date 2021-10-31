"""
作用：
（1）修改data/vocsplit文件夹中txt中的data root，并保存到voclist中
（2）重写data文件夹中txt（traindict）中的data root
"""

import argparse
import random
import os
import numpy as np
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--droot', type=str, default='/hdd1/hdd_B/bh_data/voc/')
args = parser.parse_args()

args.droot = args.droot.rstrip('/')
tgt_folder = path.join(args.droot, 'voclist')
src_folder = 'data/vocsplit' # VOC的data split。文件夹中有多个txt，每个txt表示对应class、对应shot的训练图片的文件路径


"""修改data/vocsplit文件夹中txt中的data root，并保存到voclist中"""
print('===> Converting few-shot name lists.. ')
for name_list in sorted(os.listdir(src_folder)):
    print('  | On ' + name_list)
    # Read from src 读取txt文件
    with open(path.join(src_folder, name_list), 'r') as f:
        names = f.readlines()
    
    # Replace data root 替换数据集的根文件夹
    names = [name.replace('/scratch/bykang/datasets', args.droot) for name in names]
    
    with open(path.join(args.droot, 'voclist', name_list), 'w') as f:
        f.writelines(names)


"""重写data文件夹中txt（traindict）中的data root"""
print('===> Converting class to namelist dict file ...')
for fname in ['voc_traindict_full.txt',
              'voc_traindict_bbox_1shot.txt', # txt内容为各class该shot对应的txt文件的路径
              'voc_traindict_bbox_2shot.txt',
              'voc_traindict_bbox_3shot.txt',
              'voc_traindict_bbox_5shot.txt',
              'voc_traindict_bbox_10shot.txt']: 
    full_name = path.join('data', fname)
    print('  | On ' + full_name)
    # Read lines
    with open(full_name, 'r') as f:
        lines = f.readlines()

    # Replace data root
    lines = [line.replace('/scratch/bykang/datasets', args.droot) for line in lines]

    # Rewrite lines
    with open(full_name, 'w') as f:
        f.writelines(lines)

print('===> Finished!')