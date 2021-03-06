import random
import os
from os import path

# VOC有20个class
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
		   "bus", "car", "cat", "chair", "cow", "diningtable",
		   "dog", "horse", "motorbike", "person", "pottedplant",
		   "sheep", "sofa", "train", "tvmonitor"]

# 设置5个novel class
novel_classes = ["bird", "bus", "cow", "motorbike", "sofa"]
n_novel = len(novel_classes) # novel class的数量


few_nums = [1, 2, 3, 5, 10] # k-shot的k
DROOT = '/hdd1/hdd_B/bh_data/voc' # 数据集根文件夹
PROOT = '/home/binghan/FSRW' # 项目根文件夹
root = DROOT + '/voclist/' # few-shot数据集根文件夹
outroot = PROOT + '/data' # 输出文件夹（项目）
cfgroot = PROOT + '/cfg' # config文件夹（项目）
types = ['mix', 'few']

for typ in types:
	for n in few_nums:
    	# typ：
		# n：k-shot的k

		### 该k-shot的各class的training set
		fname = 'voc_traindict_{}{}c_{}shot.txt'.format(typ, n_novel, n)
		with open(path.join(outroot, fname), 'w') as f:
			for cls_name in classes:
				if typ == 'mix' and cls_name not in novel_classes:
					f.write(cls_name + ' ' + root + '{}_train.txt\n'.format(cls_name))
				else:
					f.write(cls_name + ' ' + root + '{}shot_{}_train.txt\n'.format(n, cls_name))

		datacfg = 'voc_learnet_{}{}c_{}shot.data'.format(typ, n_novel, n)
		with open(path.join(cfgroot, datacfg), 'w') as f:
			f.write('meta = data/' + fname + '\n')
			f.write('train = data/' + fname + '\n')
			f.write('valid = data/voc_testdict_full.txt\n')
			f.write('backup = backup/meta_{}{}c_{}shot\n'.format(typ, n_novel, n))
			f.write('gpus = 0,1,2,3')

# valid = /scratch/bykang/datasets/2007_test.txt
# valid = data/voc_testdict_full.txt
