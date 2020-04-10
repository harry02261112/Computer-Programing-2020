#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[2]:


im = cv2.imread("end_of_the_road_hw1.jpg",0)
im_pad = cv2.copyMakeBorder(im, 1, 1, 1, 1,cv2.BORDER_CONSTANT,0)
im_pad


# In[3]:


plt.figure(figsize=(12,10))
plt.imshow(im_pad, cmap = "gray")


# In[4]:


flt = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
flt


# In[5]:


im_conv = np.zeros((720,960))
for i in range(im.shape[0]-1):
    for j in range(im.shape[1]-1):
        conv_sections = np.array([
            [np.int(im_pad[i][j]), np.int(im_pad[i][j+1]), np.int(im_pad[i][j+2])],
            [np.int(im_pad[i+1][j]), np.int(im_pad[i+1][j+1]), np.int(im_pad[i+1][j+2])],
            [np.int(im_pad[i+2][j]), np.int(im_pad[i+2][j+1]), np.int(im_pad[i+2][j+2])],
        ])
        im_conv[i][j] = np.int((conv_sections*flt).sum())
im_conv


# In[6]:


plt.figure(figsize=(12,10))
plt.imshow(im_conv, cmap = "gray")


# In[7]:


cv2.imwrite("im_conv.png",im_conv)


# In[ ]:




