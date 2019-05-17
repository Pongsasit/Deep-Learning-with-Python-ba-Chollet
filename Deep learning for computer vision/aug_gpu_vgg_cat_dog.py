# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:51:55 2019

@author: pongsasit
"""
#try to use GPU
import matplotlib.pyplot as plt
import os
#import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model

from keras.applications import VGG16
conv_base = VGG16(weights = 'imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))

base_dir = 'C:/Users/pongsasit/Desktop/code/cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir =os.path.join(base_dir,'test')



model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False


#augmentation

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size = 20 ,
    class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size = 20 ,
    class_mode = 'binary')

model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=2e-5),
             metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1)
plt.plot(epochs, loss, 'bo',label = 'TRAINING loss')
plt.plot(epochs,val_loss,'r',label = 'Validation loss')
plt.title('Training & Validation loss')
plt.legend()

plt.show()

model.save('C:/Users/pongsasit/Desktop/code/augmentation&vgg30epoch.h5')

