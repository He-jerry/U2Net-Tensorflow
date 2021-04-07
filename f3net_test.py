import time

import numpy as np
import cv2
import os
from keras.layers import Input
from model import VGG16
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.models import load_model

from reimp import f3net


def padding(x):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    return temp_x

def load_image(path):
    x = cv2.imread(path)
    sh = x.shape
    x = np.array(x, dtype=np.float32)
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    x = padding(x)
    x = cv2.resize(x, target_size, interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(x,0)
    return x,sh

def cut(pridict,shape):
    h,w,c = shape
    size = max(h, w)
    pridict = cv2.resize(pridict, (size,size))
    paddingh = (size - h) // 2
    paddingw = (size - w) // 2
    return pridict[paddingh:h + paddingh, paddingw:w + paddingw]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getres(pridict,shape):
    for i in range(len(pridict)):
        pridict = sigmoid(pridict)
        #print(pridict)
        pridict[i][pridict[i] > 0.5] = 1
        pridict[i][pridict[i] < 0.5] = 0
        pridict[i] = np.array(pridict[i] * 255, dtype=np.uint8)
        #pridict[i] = np.squeeze(pridict[i])
        #pridict[i] = cut(pridict[i], shape)
    return pridict

def laplace_edge(x):
    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(x/255.,-1,laplace)
    edge = np.maximum(np.tanh(edge),0)
    edge = edge * 255
    edge = np.array(edge, dtype=np.uint8)
    return edge
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = '/F3Net_keras/00050.h5'

target_size = (384,384)


model_input = Input(shape=(target_size[0],target_size[1],3))
model=f3net(model_input)
model.load_weights(model_name,by_name=True)
#model=load_model("pyramid.hdf5")
timecount=time.time()
print("Inf start time:",timecount)
total=timecount
for layer in model.layers:
    layer.trainable = False
g = os.walk(r"test/image")
for path, dir_list, file_list in g:
   for file_name in file_list:
          image_path = path+'/'+file_name
          img, shape = load_image(image_path)
          img = np.array(img, dtype=np.float32)
          sa = model.predict(img)
          sa=sa[-1]
          sa = getres(sa, shape)
          cv2.imwrite("Result/2nd/F3Net/Keras/"+file_name,sa[-1])