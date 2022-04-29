# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:49:34 2022

@author: krist
"""

import scipy.io
import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


#set working directory to folder that file is in
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



#Valid image types
valid_image=['.jpg', '.gif', '.png']

#Folder paths needed to extract images
set1paths=['\Brain Tumor','\Healthy']
set2paths=['\\yes','\\no']
testimg=[]

tum1=4508
tum2=3000

def downloadimages(paths,valid_images):
    tumor=[]
    healthy=[]
    # if num==1:
    #     rands20r=np.random.randint(0,4508, 901)
    #     rands20t=np.random.randint(0,4508, 901)
    # else:
    #     rands20r=np.random.randint(0,3000, 600)
    #     rands20t=np.random.randint(0,3000, 600)
    for ind,name in enumerate(paths):
        for ind2, f in enumerate(os.listdir(dname+name)):
            ext= os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img=(Image.open(os.path.join(dname+name,f)))
            
            #convert all images to greyscale
            img=ImageOps.grayscale(img)
            
            testimg.append(img)
            
            #randomly rotation and translating
            # if ind2 in rands20r:
            #     img=rotate(img)
            # if ind2 in rands20t:
            #     img=translate(img)
            
            #Resizing all images to be the same size
            # if images are square, just resize
            if img.size[0]==img.size[1]:
                img=img.resize((64,64))
            
            #if images are rectangle, resize and crop
            else:
                img=resizecrop(img)
                
            #convert to array for further processing
            img=np.asarray(img)

            # whiten/blacken
            img= Whiten_White_Blacken_Black(img)
            
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

def Whiten_White_Blacken_Black(img):

    NOISE_FLOOR = 20;
    WHITE_CEILING = 235;
    
    img=img.flatten()

    img[img > WHITE_CEILING] = 255
    img[img < NOISE_FLOOR] = 0
    return img.reshape(64,64)
    
def convertgrey(img):
    #rgb conversions
    rgb=np.array([0.299,0.587,0.114])
    return np.dot(img[...,:3], rgb)


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


def rotate(img):
    img_r = img.rotate(90, Image.NEAREST, expand = 1)
    return img_r

def translate(img):
    a = 1
    b = 0
    c = 5 #left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = 5 #up/down (i.e. 5/-5)
    img_t = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    return img_t

def main():
    # %% Downloading both data sets and storing in arrays

    tumor1, healthy1=downloadimages(set1paths, valid_image)
    tumor2, healthy2=downloadimages(set2paths, valid_image)
            
        # %% classifier function
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.naive_bayes import GaussianNB


    first=[tumor1,healthy1]

    second=[tumor2,healthy2]

    classifiers=[GaussianNB(var_smoothing=10**-3),LogisticRegression(max_iter=500, solver='newton-cg')]

    def run_clfs(classifier):
        for ind, item in enumerate([first, second]):
            tumor=np.ones(len(item[0]))
            notumor=np.zeros(len(item[1]))


            together=np.concatenate((item[0], item[1]))
            togethery=np.concatenate((tumor,notumor)).reshape(-1,1)

            data=np.append(together, togethery, axis=1)

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
            datax=data[:,:4096]
            datay=data[:,4096]

            #creating classifier off of first dataset
            if ind==0:
                clf=classifier.fit(xtrain,ytrain)
                print(classifier)
                print('Classification Report for first dataset')
                preds=clf.predict(xtest)
                print(confusion_matrix(ytest, preds))
                print(classification_report(ytest,preds))
            #predicts outcome of dataset2 
            if ind==1:
                preds2=clf.predict(datax)
                print(classifier)
                print('Classification Report for second dataset')
                print(confusion_matrix(datay, preds2))
                print(classification_report(datay,preds2))

    for clfs in classifiers:
        run_clfs(clfs)
    


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

