# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout,Input,Flatten,GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import uuid
from keras.utils import multi_gpu_model
import numpy as np
import glob as glob
import random
import shutil
import cv2
import heapq

def preprocess(imgpath,image_size):
	img = cv2.imread(imgpath)
	width, height, channels = img.shape
	if channels == 2:
		img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	elif channels == 3 or channels == 4:
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	minL = min(width, height)
	maxL = max(width, height)
	index = (maxL - minL) // 2
	if width >= height:
		img_crop = img_rgb[index:index + minL, :minL]
	else:
		img_crop = img_rgb[:minL, index:index + minL]
	img_resize = cv2.resize(img_crop, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
	return (img_resize.astype(np.float32) - 127.5) / 127.5

if __name__ == '__main__':

	image_size=299
	loadmode='weights'
	weights_name=''
	model_name=''
	accuracys=[]
	imgrecord=[]
	scorerecord=[]
	dataset_path='/data/dataset/scene_data_ours/Val/'
	lable_val_Dir=['sky_365/','oceanANDwave_365/','forest_365/','farm_365/','beach_365/','mountain_365/','night/','pet/','word/','green/','food_dessert/','food_others/','ice_365/','people/','sun/','street/','perform/']

	for lable in range(len(lable_val_Dir)):
		accuracy=0
		ture_val_scores=[]
		val_image_paths=[]
		outdir='./wrongIMG/'+str(lable)+'_'+lable_val_Dir[lable]
		os.makedirs(outdir)
		val_=sorted(glob.glob(os.path.join(dataset_path,lable_val_Dir[lable], '*.jpg')))
		for img_val in val_:
			val_image_paths.append(img_val)
			ture_val_scores.append(lable)
		#to numpy 
		val_image_paths = np.array(val_image_paths)
		ture_val_scores = np.array(ture_val_scores, dtype='float32')

		if loadmode =='weights':
			base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False, pooling='avg')
			x = Dropout(0.1)(base_model.output)
			x = Dense(128, activation='relu')(x)
			x = Dense(17, activation='softmax')(x)
			model = Model(base_model.input, x)
			lr=1e-5
			optimizer = Adam(lr=lr)
			model.compile(optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
			model.load_weights('/data/scene/scene_classification_res_gpu/weights/scene_InRes.01-1.33.h5')
		if loadmode =='model':
			model = load_model('/data/scene/scene_classification_res_1dense/weights/scene_InRes_Dropout0.00_Lr1.00e-05_Densen1_01-2.87.h5')
		for index in range(len(val_image_paths)):
			imgpath=val_image_paths[index]
			ture_score=ture_val_scores[index]
			img=preprocess(imgpath,image_size)
			scores = model.predict(np.array([img]))[0]
			top_number=1
			top_key=heapq.nlargest(top_number, range(len(scores)), scores.take)
			print('---top_key',top_key)
			top_value=heapq.nlargest(top_number, scores)
			print('---top_value',top_value)
			if  ture_score in top_key:
				accuracy+=1
			else:
				basename=os.path.basename(imgpath)
				basename=basename.split('.')[0].split('_')[0]+'_pre_'+str(top_key[0])+'_f_'+str(top_value[0])+'.jpg'
				outpath=os.path.join(outdir,basename)
				shutil.copy(imgpath,outpath)
			imgrecord.append(imgpath)
			scorerecord.append(scores)
		accuracys.append(accuracy/len(val_image_paths))

	imgrecord=np.array(imgrecord)
	scorerecord=np.array(scorerecord)
	accuracys=np.array(accuracys)
	record = np.vstack((imgrecord,scorerecord))
	print('---the accuracys is :', accuracys)
	np.save('./record.npy',record)
	np.save('./accuracys.npy',accuracys)
