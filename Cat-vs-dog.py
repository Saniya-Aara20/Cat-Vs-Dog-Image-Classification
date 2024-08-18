#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!pip install opencv-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


# In[9]:


directory=r"D:\machine_learning\dogscat\catdogs"
categories=['cats','dogs']


# Go to every image, convert that into array and save in list(os)

# In[10]:


IMG_SIZE=100
data=[]
for category in categories:
    folder=os.path.join(directory,category)   #concatenates the path of each category in that directory
    #print(folder)
    label=categories.index(category)         #label here is given according to the index of the category present in directory(cat-0, dog-1)
    for img in os.listdir(folder):          #images in present folder: each image is given a unique path.
        img_path=os.path.join(folder,img)
        #print(img_path)
        img_arr=cv2.imread(img_path)       #image is converted into pixels and stored in an array with numerical values corresponding to those images.
        #plt.imshow(img_arr[4])
        img_arr=cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
        data.append([img_arr,label])


# In[11]:


plt.imshow(img_arr)


# In[12]:


random.shuffle(data)
data[1]


# In[13]:


x=[]
y=[]
for features,labels in data:
    x.append(features)
    y.append(labels)
x=np.array(x)
y=np.array(y)


# In[14]:


#store them to our device
pickle.dump(x,open('x.pkl','wb'))
pickle.dump(y,open('y.pkl','wb'))


# In[15]:


#Training modelling
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard
Name=f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard=TensorBoard(log_dir=f'logs\\{Name}\\')


x=pickle.load(open('x.pkl','rb'))
y=pickle.load(open('y.pkl','rb'))
x


# In[16]:


#values are larger so reduce them to 0 to 1 scale:
x=x/225
x.shape


# In[17]:


#!pip install tensorflow
#fit model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model=Sequential()
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))


# In[19]:


#loss function
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=5,validation_split=0.1)


# In[20]:


from tensorflow.keras.callbacks import TensorBoard
Name='cat-vs-dog-prediction-{int{time.time()}}'
tb=TensorBoard(log_dir=f'logs\\{Name}\\')
model.fit(x,y,epochs=5,validation_split=0.1,batch_size=32, callbacks=[tb])

