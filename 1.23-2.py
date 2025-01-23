from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/' , 'test/']
for subdir in subdirs:
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
seed(1)
val_rate = 0.25
src_directory = 'train/'
for file in listdir(src_directory):
    src = src_directory + file
    dst_dir = 'train/'
    if random() < val_rate:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dogs'):
       dst = dataset_home + dst_dir + 'dogs/' + file
       copyfile(src, dst)
