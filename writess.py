import os
import cv2
import numpy as np
import shutil
g = os.walk(r"/home/mia_dev/Documents/dataset/OCT_Aw/train/imagere")
#g = os.walk(r"/home/mia_dev/Documents/dataset/OCT_Aw/train/image")
name=[]
f=open("train_pair.txt",'w')
f=open("train_pair.txt",'a')
for path,dir_list,file_list in g:
    for file_name in file_list:
        print(file_name)
        gt="/home/mia_dev/Documents/dataset/OCT_Aw/train/maskre"+'/'+file_name.split('.')[0]+'.png'
        print(gt)
        #shutil.copy(path+'/'+file_name,"/home/mia_dev/Documents/dataset/OCT_Aw/train/imagere/"+file_name.replace(' ','_'))
        #shutil.copy(gt,"/home/mia_dev/Documents/dataset/OCT_Aw/train/maskre/" + file_name.split('.')[0].replace(' ', '_')+'.png')
        f.write(path+'/'+file_name)
        f.write(' ')
        f.write(gt)
        f.write('\n')