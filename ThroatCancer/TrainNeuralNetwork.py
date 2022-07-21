
import os
import PIL
import glob
import time
import shutil
import random
# import imageio
# from google.colab import files
import zipfile
import numpy as np
import tensorflow as tf
# from IPython import display
# from google.colab import files
# from ThroatCancerDetection import models
from keras import regularizers
from tensorflow.keras import layers
from keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target=['Cancer', 'No Cancer']
train_count=11
test_count=8

total_count = train_count + test_count
print("Total count of images : ", total_count)

path = r'C:\\Users\\kudhroli\\Desktop\\Projects\\ThroatCancer\\ThroatCancer\\static\\styles\\images'

train_path = os.path.join(path, "Train")
print("Your training dataset path : ", train_path)
print("     ==========================             ")
test_path = os.path.join(path, "Validation")
print("Your test dataset path : ", test_path)

train_data_cancer_dir = os.path.join(train_path, "Cancer")
train_data_nonCancer_dir = os.path.join(train_path, "Non-Cancer")


test_data_cancer_dir = os.path.join(test_path, "Cancer")
test_data_nonCancer_dir = os.path.join(test_path, "Non-Cancer")

train_data_cancer_fnames = os.listdir(train_data_cancer_dir)
train_data_nonCancer_fnames = os.listdir(train_data_nonCancer_dir)

print('Total Cancer training images :', len(os.listdir( train_data_cancer_dir ) ))
print('Total Non-Cancer training images :', len(os.listdir( train_data_nonCancer_dir ) ))

print('\n\n\n')

print('Total Cancer validation images :', len(os.listdir( test_data_cancer_dir ) ))
print('Total Non-Cancer validation images :', len(os.listdir( test_data_nonCancer_dir ) ))

numRows = 5
numCols = 4
picIndex = 0

model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(4, (3,3), activation='relu',kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
 tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
 tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
 tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics = ['acc'])

train_datagen = ImageDataGenerator( rescale = 1.0/255.,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True )
test_datagen = ImageDataGenerator( rescale = 1.0/255.)
train_generator = train_datagen.flow_from_directory(train_path,batch_size=20,class_mode='binary',target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory(test_path,batch_size=20,class_mode='binary',target_size=(150, 150))

history = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=1, epochs=200, validation_steps=1, verbose=2)
model.save('h5.model')

# uploaded=files.upload()
# for fn in uploaded.keys():
#     path='/content/' + fn
#     img=image.load_img(path, target_size=(150, 150))
#     x=image.img_to_array(img)
#     x=np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
    
#     print(classes[0])
#     if classes[0]>0:
#         print(" has cancer")
#     else:
#         print(" does not have cancer")

