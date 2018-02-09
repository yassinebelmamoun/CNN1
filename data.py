from __future__ import print_function
import os
import numpy as np

import cv2
import pandas as pd
import sys
import os
import os.path
import string
import scipy.io
import pdb
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import skimage
import skimage.measure

data_path = 'raw/'
save_path = 'save/'
image_rows = 400
image_cols = 400



def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    print(images)
    total = int(len(images) / 2)
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def preprocess(imgs, img_rows,img_cols):
    imgs_p = np.ndarray((imgs.shape[0],imgs.shape[1],img_rows,img_cols),dtype=np.uint8)
    return imgs_p

def detseg():
    out_rows= 400
    out_cols= 400 
    imgs_train = np.load('imgs_train.npy')
    imgs_train=preprocess(imgs_train, out_rows, out_cols).astype(np.float32)
    mean_image=imgs_train.mean(0)[np.newaxis,]
    imgs_train -=mean_image
    std_image=imgs_train.std(0)[np.newaxis,]
    imgs_train /=std_image

    imgs_mask_train = np.load('imgs_mask_train.npy')
    imgs_mask_train=preprocess(imgs_mask_train, out_rows,out_cols)

    np.save(save_path+'mean.npy',mean_image)
    np.save(save_path+'std.npy',std_image)
    np.save(save_path+'data.npy',imgs_train.astype(np.float32))
    print('save data')
    np.save(save_path+'mask.npy',imgs_mask_train.astype(np.bool))
    print('save mask')

def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()

    detseg()
