
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



emotion_dictionary= { 0:"Angry",1:"Disgusted", 2:"Scared",3:"Happy", 4:"Naturael",5:"Sad",6: "Suprized"}
cur_path= os.path.dirname(os.path.abspath(__file__))
emoji_dist={0: os.path.join(cur_path, "data", "emojiss", "angry.png"),
1: os.path.join(cur_path, "data", "emojiss", "disgust.png"),
2: os.path.join(cur_path, "data", "emojiss", "scared.png"),
3: os.path.join(cur_path, "data", "emojiss", "happy.png"),
4: os.path.join(cur_path, "data", "emojiss", "naturael.png"), 
5: os.path.join(cur_path, "data", "emojiss", "sad.png"),
6: os.path.join(cur_path, "data", "emojiss", "suprized.png")}

global last_frame1
last_frame1= np.zeros((480,640,3), dtype=np.uint8)
global cap1
show_text=[0]
global frame_number


def show_subject():
    cap1 = cv2.VideoCapture(r'C:\Users\kbr\Downloads\Documents\emofify\src\data\testvideo\WIN_20231228_20_06_37_Pro.mp4') #camera kullanılacaksaa 0 verilir video ise video yolu
    if not cap1.isOpened():
        print("Can't open the camera")
        
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number > length:
        exit()
    cap1.set(1, frame_number)
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600,500))
    bounding_box = cv2.CascadeClassifier('C:/Users/kbr/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dictionary[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain2.imgtk = imgtk
        lmain2.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        exit()


def show_avatar():
    
    frame2 = cv2.imread(emoji_dist[show_text[0]]) 
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk = imgtk2
    lmain3.configure(text=emotion_dictionary[show_text[0]], font=('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)



if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master=root, padx=50, bg='black')
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.pack()
    lmain3.place(x=560,y=100)
    lmain2.pack(side=LEFT)
    lmain2.place(x=500,y=200)

    root.title("video To Emoji")
    root.geometry("1480x960+100+10")
    root['bg'] = 'black'
    exitButton = Button(root, text='Quit', fg='red', command=root.destroy, font=('arial', 25, 'bold')).pack(side = BOTTOM)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()



