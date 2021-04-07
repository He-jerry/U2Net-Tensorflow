import numpy as np
import cv2
import random
import os

def padding(x,y):
    h,w,c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size,size,c))
    temp_y = np.zeros((size,size))
    temp_x[paddingh:h+paddingh,paddingw:w+paddingw,:] = x
    temp_y[paddingh:h+paddingh,paddingw:w+paddingw] = y
    return temp_x,temp_y

def random_crop(x,y):
    h,w = y.shape
    randh = np.random.randint(h/8)
    randw = np.random.randint(w/8)
    randf = np.random.randint(10)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
    if randf >= 5:
        x = x[::, ::-1, ::]
        y = y[::, ::-1]
    return x[p0:p1,p2:p3],y[p0:p1,p2:p3]

def random_rotate(x,y):
    angle = np.random.randint(-25,25)
    h, w = y.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(x, M, (w, h)),cv2.warpAffine(y, M, (w, h))

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def getTrainGenerator(file_path,mask_path, target_size, batch_size, israndom=False):
    g = os.walk(file_path)
    batch_x = []
    batch_y = []
    p=[]
    for path, dir_list, file_list in g:
        for file_name in file_list:
            p.append([path + '/' + file_name,mask_path+'/'+ file_name.split('.')[0] + '.png'])
    while True:
        for i in range(len(p)):
            img_path = p[i][0]
            mask_path = p[i][1]
            x = cv2.imread(img_path)
            y = cv2.imread(mask_path)
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if len(y.shape) == 3:
                y = y[:,:,0]
            y = y/y.max()
            if israndom:
                x,y = random_crop(x,y)
                x,y = random_rotate(x,y)
                x = random_light(x)

            x = x[..., ::-1]
            x, y = padding(x, y)

            x = cv2.resize(x, target_size, interpolation=cv2.INTER_LINEAR)
            y = cv2.resize(y, target_size, interpolation=cv2.INTER_NEAREST)
            y = y.reshape((target_size[0],target_size[1],1))
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == batch_size:
                # pred1, pred2, out2h, out3h, out4h, out5h
                yield (np.array(batch_x, dtype=np.float32), {'side1': np.array(batch_y, dtype=np.float32), 'side2': np.array(batch_y, dtype=np.float32),'side3': np.array(batch_y, dtype=np.float32),'side4': np.array(batch_y, dtype=np.float32),'side5': np.array(batch_y, dtype=np.float32),'side6': np.array(batch_y, dtype=np.float32),'out': np.array(batch_y, dtype=np.float32)})
                #yield (np.array(batch_x, dtype=np.float32), {'pred1': np.array(batch_y, dtype=np.float32), 'pred2': np.array(batch_y, dtype=np.float32),'out2h': np.array(batch_y, dtype=np.float32),'out3h': np.array(batch_y, dtype=np.float32),'out4h': np.array(batch_y, dtype=np.float32),'out5h': np.array(batch_y, dtype=np.float32)})
                #yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))
                batch_x = []
                batch_y = []