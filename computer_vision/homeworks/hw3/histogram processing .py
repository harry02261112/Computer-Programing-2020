#!/usr/bin/env python
# coding: utf-8

# In[224]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = plt.imread('mp2.jpg',"gray") # importing the image mp2.jpg
img = img.max(axis=-1) # reducing the image channels into 1


# In[28]:


# A equation merely to return the place where our desirable value is located in an array or list
def the_num(n,nums):
    for o in range(len(nums)):
        if (int(nums[o]) == n):
            return o


# In[226]:


# My equalization histogram equation written according to the equaiton on the textbook
def my_histogram(im):
    # Determine our output image. To maintain the same shape I just copy the origin on it 
    img_altered = im
    # Listing out the distinct numbers, ranging from 0 to 255, in the input image from small to big 
    # This is for acquiring the number of these distinct numbers to add up later according to the equation
    nums = list(np.unique(im))
    # Setting the accumulation of the number of the distinct numbers in an array
    sums = np.zeros(len(nums))
    # Set the first cell as the number of the smallest distinct number in the image 
    sums[0] = list(im.ravel()).count(nums[0])
    # Iterating to accumulate the number of the distinct numbers and saving it in each cells of sums
    for i in range(1,len(nums)):
        sums[i] = sums[i-1] + list(im.ravel()).count(nums[i])
        
    # This is the actual EH process 
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # k is just an indication of which the distinct number it is in the input image
            k = the_num(im[i][j],nums)
            # This is the equation on the textbook
            img_altered[i][j] = (max(im.ravel()))*(sums[k]/len(im.ravel()))
    return img_altered


# In[29]:


# applying my EH on mp2.jpg
img_altered = my_histogram(img)


# In[215]:


# normalizing the processed img to 0~255
img_altered_whole_spectrum = img_altered
for i in range(img_altered.shape[0]):
    for j in range(img_altered.shape[1]): 
        img_altered_whole_spectrum[i][j] = (img_altered[i][j]*255)/max(img_altered.ravel())


# In[255]:


# Showing the result
plt.figure(figsize=(20,10))
plt.subplot(121),plt.hist(img_altered_whole_spectrum.ravel(),max(img_altered_whole_spectrum.ravel()),[0,max(img_altered_whole_spectrum.ravel())],density = True)
plt.subplot(122),plt.imshow(img_altered_whole_spectrum,"gray")
plt.savefig('1stq_myeq.png', bbox_inches='tight')
plt.show()


# In[257]:


# Applying the cv2 EH on mp2.jpg
cv2eq = cv2.equalizeHist(img)
plt.figure(figsize=(20,10))
plt.subplot(121),plt.hist(cv2eq.ravel(),max(cv2eq.ravel()),[0,max(cv2eq.ravel())],density = True)
plt.subplot(122),plt.imshow(cv2eq, cmap = "gray")
plt.savefig('1stq_cv2eq.png', bbox_inches='tight')
plt.show()


# In[136]:


# Read mp2a.jpg
img2 = plt.imread("mp2a.jpg")
img2_cv2 = cv2.imread("mp2a.jpg")


# In[138]:


# Separating bgr channels 
b,g,r = cv2.split(img2_cv2)
rgb_img = cv2.merge([r,g,b])
plt.figure(figsize=(12,10))
plt.subplot(141), plt.imshow(rgb_img)
plt.subplot(142), plt.imshow(r)
plt.subplot(143), plt.imshow(g)
plt.subplot(144), plt.imshow(b)
plt.show()


# In[52]:


# Storing the separated channels in three seperate arrays 
img2R = np.zeros((512,512))
img2G = np.zeros((512,512))
img2B = np.zeros((512,512))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        img2R[i][j] = img2[i][j][0]
        img2G[i][j] = img2[i][j][1]
        img2B[i][j] = img2[i][j][2]


# In[53]:


plt.figure(figsize=(12,10))
plt.subplot(141), plt.imshow(img2)
plt.subplot(142), plt.imshow(img2R)
plt.subplot(143), plt.imshow(img2G)
plt.subplot(144), plt.imshow(img2B)
plt.show()


# In[54]:


# Applying my EH on the R channel
img2R_altered = my_histogram(img2R)


# In[55]:


# Applying my EH on the G channel
img2G_altered = my_histogram(img2G)


# In[56]:


# Applying my EH on the B channel
img2B_altered = my_histogram(img2B)


# In[120]:


# Recombining the EH channels
new_img2 = np.zeros((512,512,3))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        pix = [int(img2R_altered[i][j]),int(img2G_altered[i][j]),int(img2B_altered[i][j])]
        new_img2[i][j] = pix
print(new_img2)


# In[142]:


# Applying cv2 EH on the three channels respectively 
cv2eqb = cv2.equalizeHist(b)
cv2eqr = cv2.equalizeHist(r)
cv2eqg = cv2.equalizeHist(g)

# Recombining the EH channels
cv2eqrgb = np.zeros((512,512,3))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        pix = [int(cv2eqr[i][j]),int(cv2eqg[i][j]),int(cv2eqb[i][j])]
        cv2eqrgb[i][j] = pix
print(cv2eqrgb)


# In[228]:


plt.figure(figsize = (20,8))
plt.subplot(141),plt.hist(cv2eqrgb.ravel(),255,[0,255],density = True)
plt.subplot(142),plt.imshow(cv2eqrgb.astype('uint8'))
plt.subplot(143),plt.hist(new_img2.ravel(),255,[0,255],density = True)
plt.subplot(144),plt.imshow(new_img2.astype('uint8'))
plt.savefig('cv2rgbeq_and_myrgbeq.png', bbox_inches='tight')
plt.show()


# In[170]:


plt.figure(figsize = (20,10))
plt.subplot(121),plt.hist(img2.ravel(),max(img2.ravel()),[0,max(img2.ravel())],density = True)
plt.subplot(122),plt.imshow(img2.astype('uint8'))
plt.savefig('origin_mp2a.png', bbox_inches='tight')
plt.show()


# In[62]:


# Separating the HSV channels
img2_cv2 = cv2.imread("mp2a.jpg")
HSV = cv2.cvtColor(img2_cv2, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(HSV)


# In[63]:


# Applying my EH on the H channel
img2H_altered = my_histogram(H)


# In[174]:


# Applying cv2 EH on the H channel
Hcv2eq = cv2.equalizeHist(H)


# In[175]:


plt.figure(figsize = (20,10))
plt.subplot(141),plt.hist(Hcv2eq.ravel(),max(Hcv2eq.ravel()),[0,max(Hcv2eq.ravel())],density = True)
plt.subplot(142),plt.imshow(Hcv2eq.astype('uint8'))
plt.subplot(143),plt.hist(img2H_altered.ravel(),max(img2H_altered.ravel()),[0,max(img2H_altered.ravel())],density = True)
plt.subplot(144),plt.imshow(img2H_altered.astype('uint8'))
plt.savefig('cv2eqH_and_myh.png', bbox_inches='tight')
plt.show()


# In[171]:


plt.figure(figsize = (20,10))
plt.subplot(141),plt.hist(H.ravel(),max(H.ravel()),[0,max(H.ravel())],density = True)
plt.subplot(142),plt.imshow(H.astype('uint8'))
plt.subplot(143),plt.hist(img2H_altered.ravel(),max(img2H_altered.ravel()),[0,max(img2H_altered.ravel())],density = True)
plt.subplot(144),plt.imshow(img2H_altered.astype('uint8'))
plt.savefig('originh_and_myh.png', bbox_inches='tight')
plt.show()


# In[241]:


# Separating the Y channel from YCbCr
imgYcc = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
y,b,r = cv2.split(imgYcc)


# In[65]:


# Applying my EH on the Y channel
img2Y_altered = my_histogram(y)


# In[161]:


plt.figure(figsize = (20,10))
plt.subplot(141),plt.hist(y.ravel(),max(y.ravel()),[0,max(y.ravel())],density = True)
plt.subplot(142),plt.imshow(y)
plt.subplot(143),plt.hist(img2Y_altered.ravel(),max(img2Y_altered.ravel()),[0,max(img2Y_altered.ravel())],density = True)
plt.subplot(144),plt.imshow(img2Y_altered)
plt.savefig('originy_and_myy.png', bbox_inches='tight')
plt.show()


# In[172]:


# Applying cv2 EH on the y channel
ycv2eq = cv2.equalizeHist(y)


# In[173]:


plt.figure(figsize = (20,10))
plt.subplot(141),plt.hist(ycv2eq.ravel(),max(ycv2eq.ravel()),[0,max(ycv2eq.ravel())],density = True)
plt.subplot(142),plt.imshow(ycv2eq)
plt.subplot(143),plt.hist(img2Y_altered.ravel(),max(img2Y_altered.ravel()),[0,max(img2Y_altered.ravel())],density = True)
plt.subplot(144),plt.imshow(img2Y_altered)
plt.savefig('cv2eqy_and_myy.png', bbox_inches='tight')
plt.show()


# In[248]:


cv2eqrgb_for_cv2_imwrite = np.zeros((512,512,3))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        pix = [int(cv2eqb[i][j]),int(cv2eqg[i][j]),int(cv2eqr[i][j])]
        cv2eqrgb_for_cv2_imwrite[i][j] = pix


# In[249]:


new_img2_for_cv2_imwrite = np.zeros((512,512,3))
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        pix = [int(img2B_altered[i][j]),int(img2G_altered[i][j]),int(img2R_altered[i][j])]
        new_img2_for_cv2_imwrite[i][j] = pix


# In[254]:


cv2.imwrite("1stQ_myeq_write.png", img_altered_whole_spectrum)
cv2.imwrite("1stQ_cv2eq_write.png", cv2eq)
cv2.imwrite("2stQ_myrgbeq.png", new_img2_for_cv2_imwrite)
cv2.imwrite("2stQ_cv2rgbeq.png", cv2eqrgb_for_cv2_imwrite)
cv2.imwrite("2stQ_cv2Heq.png", Hcv2eq)
cv2.imwrite("2stQ_myHeq.png", img2H_altered)
cv2.imwrite("2stQ_cv2yeq.png", ycv2eq)
cv2.imwrite("2stQ_myyeq.png", img2Y_altered)

