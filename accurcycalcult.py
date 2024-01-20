

import numpy as np 
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten  
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


emotion_model= Sequential()
#
emotion_model.add(Conv2D(32 , kernel_size=(3,3), activation= 'relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64 , kernel_size=(3,3), activation= 'relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128 , kernel_size=(3,3), activation= 'relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128 , kernel_size=(3,3), activation= 'relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7,activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
# Load the model weights
emotion_model.load_weights('model.h5')

# Prepare your test data (this should be similar to how you prepared your training/validation data)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',  # Replace with your test directory
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Evaluate the model
loss, accuracy = emotion_model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")