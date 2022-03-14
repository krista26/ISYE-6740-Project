# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:49:34 2022

@author: krist
"""

import scipy.io
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#set working directory to ISYE 6740
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
rgb=np.array([0.299,0.587,0.114])

tumorlist=[]
healthylist=[]
for name in ['\Brain Tumor','\Healthy']:
    valid_image=['.jpg', '.gif', '.png']
    for f in os.listdir(dname+name):
        ext= os.path.splitext(f)[1]
        if ext.lower() not in valid_image:
            continue
        img=(Image.open(os.path.join(dname+name,f)))
        #Resizing all images to be the same size so processing is easier
        img=img.resize((64,64))
        img=np.asarray(img)
        img=np.dot(img[...,:3], rgb)
        img=img.flatten()
        if not img.shape==(64,):
            if name=='\Brain Tumor':
                tumorlist.append(img)
            else:
                healthylist.append(np.asarray(img)) 
           
# %% testing

print(tumorlist[1].shape)

# %% Converting list to array and deleting images which are the incorrect size
tumorlist=np.asarray(tumorlist)
healthylist=np.array(healthylist)
print(healthylist.shape)



# %% Playing around with a small subset
testing=tumorlist[:500]
print(testing.shape)

testinghealthy=healthylist[:500]



#%%
from sklearn.model_selection import train_test_split
tumor=np.ones(len(tumorlist))
notumor=np.zeros(len(healthylist))

together=np.concatenate((tumorlist, healthylist))
togethery=np.concatenate((tumor,notumor))

#Train test split
train, test= train_test_split(together, test_size=0.2)
print(train.shape)

#HELLO THIS IS SEAN





