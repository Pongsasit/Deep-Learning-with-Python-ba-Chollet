# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:53:47 2019

@author: pongsasit
"""

from keras.models import load_model

model = load_model('C:/Users/pongsasit/Desktop/code/augmentation&vgg30epoch.h5')

img_path = 'C:/Users/pongsasit/Desktop/code/cats_and_dogs_small/test/cats/cat.1501.jpg'
#img_path= 'C:/Users/pongsasit/Desktop/code/cats_and_dogs_small/test/dogs/dog.1501.jpg'
#img_path = 'C:/Users/pongsasit/Desktop/code/cats_and_dogs_small/test/dogs/dog.1503.jpg'
#img_path= 'C:/Users/pongsasit/Desktop/code/kaggle_original_data/dogs-vs-cats/test1/test1/218.jpg'
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path,target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)


print(img_tensor.shape)

result = model.predict(img_tensor)
#training_set.class_indices
if result[0][0] == 1:
    print ('dog')
else:
    print ('cat')

img_tensor /= 255.
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()