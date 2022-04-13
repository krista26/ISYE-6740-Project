# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:49:34 2022

@authors: 
1.	Krista Radecke (GT ID: 903736778)
2.	Sandra Kim (GT ID: 902745258)
3.	Sean Lee (GT ID: 903648883)

"""

import scipy.io
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Valid image types
valid_image=['.jpg', '.gif', '.png']

#Folder paths needed to extract images
set1paths=['\Brain Tumor','\Healthy']
set2paths=['\\yes','\\no']


def downloadimages(paths,valid_images):
    tumor=[]
    healthy=[]
    for ind,name in enumerate(paths):
        for f in os.listdir(dname+name):
            ext= os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img=(Image.open(os.path.join(dname+name,f)))
            
           #Resizing all images to be the same size
            # if images are square, just resize
            if img.size[0]==img.size[1]:
                img=img.resize((64,64))
            
            #if images are rectangle, resize and crop
            else:
                img=resizecrop(img)
                
            #convert to array for further processing
            img=np.asarray(img)
            
            #If image is in color, convert to greyscale
            if not img.shape==(64,64):
                img=convertgrey(img)
                
            #Remove white borders if they are present
            if img[0][0]>50:
                img=whiteborders(img)
                
            #Add images to a list based on if they represent tumor or not
            if ind==0:
                tumor.append(img.flatten())
            else:
                healthy.append(img.flatten()) 
    #Convert output lists into arrays            
    tumor=np.asarray(tumor)
    healthy=np.asarray(healthy)
    return tumor, healthy

#function to remove white borders if the pic has one
def whiteborders(img):
    cop=img.copy()
    if np.mean(cop[0,:])>50:
        cop[0,:]=0
    if np.mean(cop[-1,:])>50:
        cop[-1,:]=0
    if np.mean(cop[:,0])>50:
        cop[:,0]=0
    if np.mean(cop[:,-1])>50:
        cop[:,-1]=0
    return cop
def resizecrop(img):
    side=64
    shortside=np.argmin(np.array(img).shape[:2])
    #np array switched the indexes, so 1 is the index of the width
    if shortside==1:
        #converting the width to 64, keeping aspect ratio of width
        wpercent = (side/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((side,hsize), Image.ANTIALIAS)
        #calculate the extra pixels on either side of image
        leftover=(img.size[1]-img.size[0])//2
        w,h=img.size
        #crop extra pixels off
        img=img.crop((0,leftover,w,h-leftover))
    else:
        #converting height to 64, keeps aspect ratio for height
        hpercent=(side/float(img.size[1]))
        wsize=int((float(img.size[0])*float(hpercent)))
        img=img.resize((wsize,side),Image.ANTIALIAS)
        #calculate extra pixels on either side
        leftover=(img.size[0]-img.size[1])//2
        w,h=img.size
        #crop extra pixels off
        img=img.crop((leftover,0,w-leftover,h))
        
    #failsafe resize just incase there were odd numbers
    if not img.size==(64,64):
        img=img.resize((64,64))
    return img

def convertgrey(img):
    #rgb conversions
    rgb=np.array([0.299,0.587,0.114])
    return np.dot(img[...,:3], rgb)

def main():
    # %% Downloading both data sets and storing in arrays

    tumor1, healthy1=downloadimages(set1paths, valid_image)
    tumor2, healthy2=downloadimages(set2paths, valid_image)

                    
            
    # %% testing

    print(len(tumor1))
    print(len(healthy1))
    print(len(tumor2))
    print(len(healthy2))




    #%% Concatenating data sets, shuffling data, and creating test/train splits
    from sklearn.model_selection import train_test_split
    tumor=np.ones(len(tumor1))
    notumor=np.zeros(len(healthy1))

    together=np.concatenate((tumor1, healthy1))
    togethery=np.concatenate((tumor,notumor)).reshape(-1,1)

    print(togethery.shape)
    print(together.shape)

    data=np.append(together, togethery, axis=1)
    print(data.shape)

    #Shuffle data
    np.random.seed(2)
    shuff=np.random.permutation(len(data))
    data=data[shuff,:]

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
    #Comment/uncomment to switch data set
    dataset=tumor1
    #dataset=tumor2

    f, axarr = plt.subplots(1,4)
    rands=np.random.randint(0,len(dataset),4)
    axarr[0].imshow(dataset[rands[0]].reshape(64,64), cmap='gray')
    axarr[1].imshow(dataset[rands[1]].reshape(64,64),cmap='gray')
    axarr[2].imshow(dataset[rands[2]].reshape(64,64),cmap='gray')
    axarr[3].imshow(dataset[rands[3]].reshape(64,64),cmap='gray')
    plt.suptitle("4 Random Brains with Tumors", y=.7)
    plt.subplots_adjust(wspace=0.4)

    # %% Plotting 4 random healthy brains
    #Comment/uncomment to switch data set
    #healthyset=healthy1
    healthyset=healthy2
    f, axarr = plt.subplots(1,4)
    rands=np.random.randint(0,len(healthyset),4)
    axarr[0].imshow(healthyset[rands[0]].reshape(64,64), cmap='gray')
    axarr[1].imshow(healthyset[rands[1]].reshape(64,64),cmap='gray')
    axarr[2].imshow(healthyset[rands[2]].reshape(64,64),cmap='gray')
    axarr[3].imshow(healthyset[rands[3]].reshape(64,64),cmap='gray')
    plt.suptitle("4 Random Brains without Tumors", y=.7)
    plt.subplots_adjust(wspace=0.4)

if __name__ == "__main__":
    main()

