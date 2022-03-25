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
togethery=np.concatenate((tumor,notumor)).reshape(-1,1)

print(togethery.shape)

data=np.append(together, togethery, axis=1)
print(data.shape)

#Train test split
train, test= train_test_split(data, test_size=0.2)
xtrain=train[:,:4096]
ytrain=train[:,4096]
xtest=test[:,:4096]
ytest=test[:,4096]

# %% Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logreg=LogisticRegression(max_iter=500, solver='newton-cg').fit(xtrain,ytrain)
print('Logistic Regression classification Report')
preds=logreg.predict(xtest)
print(confusion_matrix(ytest, preds))
print(classification_report(ytest,preds))


# %% Plotting 4 random tumors

f, axarr = plt.subplots(1,4)
rands=np.random.randint(0,len(tumorlist),4)
axarr[0].imshow(tumorlist[rands[0]].reshape(64,64), cmap='gray')
axarr[1].imshow(tumorlist[rands[1]].reshape(64,64),cmap='gray')
axarr[2].imshow(tumorlist[rands[2]].reshape(64,64),cmap='gray')
axarr[3].imshow(tumorlist[rands[3]].reshape(64,64),cmap='gray')
plt.suptitle("4 Random Brains with Tumors", y=.7)
plt.subplots_adjust(wspace=0.4)

# %% Plotting 4 random healthy brains
f, axarr = plt.subplots(1,4)
rands=np.random.randint(0,len(healthylist),4)
axarr[0].imshow(healthylist[rands[0]].reshape(64,64), cmap='gray')
axarr[1].imshow(healthylist[rands[1]].reshape(64,64),cmap='gray')
axarr[2].imshow(healthylist[rands[2]].reshape(64,64),cmap='gray')
axarr[3].imshow(healthylist[rands[3]].reshape(64,64),cmap='gray')
plt.suptitle("4 Random Brains without Tumors", y=.7)
plt.subplots_adjust(wspace=0.4)





