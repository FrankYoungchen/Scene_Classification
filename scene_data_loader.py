# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob

import tensorflow as tf
from tqdm import tqdm
import random
from scipy import misc

dataset_path = "/data/dataset/scene_data_ours/"
IMAGE_SIZE = 299
train_image_paths = []
train_scores = []
val_image_paths = []
val_scores = []

label_type=['sky_365','oceanANDwave_365','forest_365','farm_365','beach_365','mountain_365','night','pet','word','green','food_dessert','food_others','ice_365','people','sun','street','perform']
lable_train_Dir=['Train/sky_365/','Train/oceanANDwave_365/','Train/forest_365/','Train/farm_365/','Train/beach_365/','Train/mountain_365/','Train/night/','Train/pet/','Train/word/','Train/green/','Train/food_dessert/','Train/food_others/','Train/ice_365/','Train/people/','Train/sun/','Train/street/','Train/perform/']
lable_val_Dir=['Val/sky_365/','Val/oceanANDwave_365/','Val/forest_365/','Val/farm_365/','Val/beach_365/','Val/mountain_365/','Val/night/','Val/pet/','Val/word/','Val/green/','Val/food_dessert/','Val/food_others/','Val/ice_365/','Val/people/','Val/sun/','Val/street/','Val/perform/']
for lable in range(len(label_type)):
    train_=sorted(glob.glob(os.path.join(dataset_path,lable_train_Dir[lable], '*.jpg')))
    for img in train_:
        train_image_paths.append(img)
        train_scores.append(lable)
    val_=sorted(glob.glob(os.path.join(dataset_path,lable_val_Dir[lable], '*.jpg')))
    for img_val in val_:
        val_image_paths.append(img_val)
        val_scores.append(lable)

train_data=list(zip(train_image_paths,train_scores))
val_data=list(zip(val_image_paths,val_scores))
random.shuffle(train_data)
random.shuffle(val_data)
train_data= list(zip(*train_data))
val_data= list(zip(*val_data))
train_image_paths=train_data[0]#tuple
train_scores=train_data[1]#tuple
val_image_paths=val_data[0]#tuple
val_scores=val_data[1]#tuple

#to numpy 
train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')
val_image_paths = np.array(val_image_paths)
val_scores = np.array(val_scores, dtype='float32')
#one hot
train_scores=tf.keras.utils.to_categorical(train_scores,num_classes=None)
val_scores=tf.keras.utils.to_categorical(val_scores,num_classes=None)

size_train=len(train_image_paths)
size_val=len(val_image_paths)

print('Train set size : ', train_image_paths.shape, train_scores.shape)
print('Val set size : ', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready !')

def parse_data(filename, scores):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.resize_images(image, (256, 256))
    shape = tf.shape(image)
    minimum=tf.minimum(shape[0],shape[1])
    image = tf.random_crop(image, size=(minimum, minimum, 3))
    angle = tf.random_uniform((1,),minval=-30,maxval=30,dtype=tf.float32)
    image = tf.contrib.image.rotate(image,angle[0],interpolation='BILINEAR')
    image = tf.image.resize_images(image, size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    shape = tf.shape(image)
    minimum = tf.minimum(shape[0],shape[1])
    maximum = tf.maximum(shape[0], shape[1])
    index = (maximum - minimum) // 2
    image = tf.cond(shape[0]>=shape[1], lambda: tf.image.crop_to_bounding_box(image, index,0,minimum,minimum), lambda: tf.image.crop_to_bounding_box(image,0,index,minimum,minimum))
    image = tf.image.resize_images(image, size=(IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    with tf.Session() as sess:
        # create a dataset
        print('----type(train_image_paths)',type(train_image_paths))#if numpy
        print('----len train_image_paths',len(train_image_paths))
        print('----train_image_paths',train_image_paths)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=32)
        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()
        train_batch = train_iterator.get_next()
        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def val_generator(batchsize):
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)
        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()
        val_batch = val_iterator.get_next()
        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)











