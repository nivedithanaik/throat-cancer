from django.shortcuts import render, redirect, HttpResponse
from django.contrib import messages
from .models import FilesAdmin
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import django

from werkzeug.utils import secure_filename


import os
import PIL
import glob
import time
import shutil
import random
import imageio
import zipfile
import numpy as np
import tensorflow as tf
from keras.models import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from keras.layers import *
from keras.models import *
# from IPython import display
# from google.colab import files
from keras import regularizers
from tensorflow.keras import layers
from keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def trainNeuralNetwork(train_path, test_path):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (3,3), activation='relu',kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(8, (3,3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    #  tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #  tf.keras.layers.MaxPooling2D(2,2),
    #  tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #  tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics = ['acc'])
    train_datagen = ImageDataGenerator( rescale = 1.0/255.,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True )
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )
    train_generator = train_datagen.flow_from_directory(train_path,batch_size=20,class_mode='binary',target_size=(150, 150))
    validation_generator = test_datagen.flow_from_directory(test_path,batch_size=20,class_mode='binary',target_size=(150, 150))
    history = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=1, epochs=20, validation_steps=1, verbose=2)
    model.save('h5.model')
    print('Saved the model')

def fileUpload(f):
    with open('C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/static/styles/images/UploadedImages/'+f.name, 'wb+') as destination:
        for chunk in f.chunks():  
            destination.write(chunk)

# Create your views here.
def home(request):

    print(request)

    if request.method == 'POST':
        f = request.FILES['upload']
        path = default_storage.save('image.jpg', ContentFile(f.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)

        print('-----------------------------')

        print(tmp_file)

        
        img = tf.keras.utils.load_img(tmp_file, target_size=(150, 150))
        
        # shutil.move(r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/media/image.jpg", r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/UploadedImages/")
        # os.rename(saveImg, r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/UploadedImages/image" + str(num) + ".jpg")
        
        # x=image.img_to_array(img)
        x=tf.keras.utils.img_to_array(img)

        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
        model = load_model('h5.model')
        classes = model.predict(images, batch_size=10)
        # img.save("C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/static/styles/images/UploadedImages/dolls.jpg")
        print(classes[0])
        if classes[0]>0:
            messages.info(request, 'CANCER DIAGNOSED')
            messages.info(request, 'There is a ' + str(int(classes[0]*100*1.5)) + '"%" probability that cancer exists. Please consult doctor immediately.')
            if classes[0] > 0 and classes[0] <= 0.2:
                messages.info(request, 'Stage 1 tumour found. Size of tumour: ' + str(round(float(classes[0]*10),2)))
            elif classes[0] > 0.2 and classes[0] <= 0.5:
                messages.info(request, 'Stage 2 tumour found. Size of tumour: ' + str(round(float(classes[0]*10),2)))
            elif classes[0] > 0.5 and classes[0] <= 0.9:
                messages.info(request, 'Stage 3 tumour found. Size of tumour: ' + str(round(float(classes[0]*10),2)))
            else:
                messages.info(request, 'Stage 4 tumour found. Size of tumour: ' + str(round(float(classes[0]*10),2)))
            # trainImg = r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/UploadedImages/image" + str(num) + ".jpg" 
            # shutil.move(trainImg, r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/static/styles/images/Train/Cancer/")
        else:
            messages.error(request, 'NO CANCER DIAGNOSED')
            # trainImg = r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/UploadedImages/image" + str(num) + ".jpg" 
            # shutil.move(trainImg, r"C:/Users/kudhroli/Desktop/Projects/ThroatCancer/ThroatCancer/static/styles/images/Train/Non-Cancer/")
        return render(request, 'index.html')
    else:
        return render(request, 'index.html')