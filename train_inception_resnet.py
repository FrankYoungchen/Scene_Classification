# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.models import Model
from keras.layers import Dense, Dropout,Input,Flatten,GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import uuid
from keras.utils import multi_gpu_model
from scene_data_loader import train_generator, val_generator, size_train, size_val

path='/data/scene/scene_classification'
weights_Dir=os.path.join(path,'weight')
log_Dir=os.path.join(path,'log')

weight_name='vio_incresv2_0612_val_4000_finetune'
weight_path=os.path.join(weights_Dir,weight_name)
log_path=os.path.join(log_Dir,weight_name)

image_size = 299

img_dim=(image_size,image_size,3)

base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False, pooling='avg')
# print(base_model.layers)
for lay in base_model.layers:
    lay.trainable = True 
# for lay in base_model.layers[-4:]:
    # lay.trainable = True #fintune
x = Dropout(0.1)(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(17, activation='softmax')(x) # 0 means N, 1 means P
model = Model(base_model.input, x)
optimizer = Adam(lr=1e-5)
model.compile(optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# resuming
# if os.path.exists('weights/vio_incresv2_0612_val_4000.01-0.37.h5'):
#     model.load_weights('weights/vio_incresv2_0612_val_4000.01-0.37.h5')
checkpoint = ModelCheckpoint('weights/scene_InRes.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,mode='min')
#checkpoint = ModelCheckpoint('weights/lego_mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
logname = "./logs/scene_InRes%s" % str(uuid.uuid4())
os.makedirs(logname)
tensorboard = TensorBoard(log_dir=logname)


#reducelr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, verbose=1, mode='auto', epsilon=0.002, min_lr=1e-8)
callbacks = [tensorboard, checkpoint]

batchsize = 64
epochs = 100
last_epoch = 0
# class_weight = {0 : 1.2,
#     1: 1.}#sample imbalance
model.fit_generator(train_generator(batchsize=batchsize),
                    steps_per_epoch=(size_train // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(batchsize=batchsize),
                    validation_steps=(size_val // batchsize),
                    initial_epoch=last_epoch)


