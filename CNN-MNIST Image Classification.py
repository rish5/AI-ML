#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
import tensorflow as tf
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[9]:


## Download Image Data
(train_digits, train_labels), (test_digits, test_labels) = load_data(path = 'mnist.npz')


# In[10]:


train_digits[0]


# In[5]:


# 60000 is the no. of samples, shape of image = 28 * 28(height,width)
train_digits.shape


# In[11]:


test_digits.shape


# In[13]:


train_labels[0]


# In[18]:


plt.figure(figsize=(20,4))
# zip is used to form a pair of image data & labels
# enumerate will ouput as index values
# reshaping is done but not for this image but written for the concept
for index, (image,label) in enumerate(zip(train_digits[1:6], train_labels[1:6])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (28,28)), cmap='gray')
    plt.title("Label %i" %label, fontsize = 15)


# ### Input Shape of Image

# In[20]:


image_height = train_digits.shape[1]
image_width = train_digits.shape[2] 

# grayscale image is having rank = 1
num_channels = 1

train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))


# In[21]:


train_digits.shape[0]


# In[22]:


train_data.shape


# In[23]:


test_data = np.reshape(test_digits, (test_digits.shape[0], image_height, image_width, num_channels))


# In[24]:


test_data.shape


# ## Label or Target

# In[26]:


pd.Series(train_labels).unique()


# ### Reshaping of target into categorical data

# In[28]:


# for multi-class use one-hot encoding
from keras.utils import to_categorical
num_classes = 10
train_labels_cat = to_categorical(train_labels, num_classes)
test_labels_cat = to_categorical(test_labels, num_classes)


# In[30]:


# just for checking
train_labels_cat[5]


# In[31]:


train_labels[5]


# ### Image Normalization

# In[33]:


train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32')/255


# ### Split Data into Train and Validation

# In[34]:


from sklearn.model_selection import train_test_split
train_data2, val_data, train_labels2, val_label = train_test_split(train_data, train_labels_cat, test_size=0.1, random_state=2)


# In[36]:


train_data2.shape


# In[37]:


val_data.shape


# ### CNN - Convolutional Neural Network

# In[39]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[40]:


def build_model():
    model = Sequential()
    # Input Layer 
    # Conv Net(units , kernel shape (3,3), activation = 'ReLU', zero Padding) + input shape (height, width, num_channels)
    model.add(Conv2D(filters= 64, kernel_size= (3,3), activation='relu', padding='same', 
                    input_shape = (image_height, image_width, num_channels )))
    
    ## Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters= 64, kernel_size= (3,3), activation='relu', padding='same', 
                    input_shape = (image_height, image_width, num_channels )))
    
    ## Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters= 64, kernel_size= (3,3), activation='relu', padding='same', 
                    input_shape = (image_height, image_width, num_channels )))
    
    ## Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    ## Flatten
    model.add(Flatten())
    
    ## Dense Layer / Fully Connected Layer
    model.add(Dense( units= 128, activation= 'relu'))
    ## Output Layer 
    model.add(Dense(units= num_classes, activation='softmax'))
    
    
    ## Optimizers & Loss Function
    # Loss Function - CrossEntropy
    ## Binary Class - 'binary_crossentropy', 'multiclass' :'categorical_crossentropy'
    # -plog(p) - p log(1-p) where p is predicted o/p given by hypothesis equation 
    model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    
    return model
    


# In[41]:


model = build_model()


# In[42]:


model.summary()


# In[43]:


history = model.fit(train_data2, train_labels2, epochs=15, batch_size=64, validation_data=(val_data, val_label))


# In[44]:


import pandas as pd
pd.DataFrame(history.history)


# In[45]:


sns.set_style('darkgrid')
pd.DataFrame(history.history).plot()


# In[54]:


predicted_val = model.predict(test_data)
predicted_val


# ## Classification Report

# In[65]:


from sklearn.metrics import confusion_matrix
confusion_matrix(np.argmax(test_labels_cat, axis=1), np.argmax(predicted_val, axis=1))


# In[60]:


import cv2


# In[67]:


img = cv2.imread('C:/Users/RISHI/Desktop/2.png',0)


# In[68]:


img.shape


# In[69]:


img = cv2.resize(img, (28, 28))


# In[70]:


img = cv2.bitwise_not(img)
train_data.shape


# In[71]:


# New Image
image_data = img.reshape(1, 28 , 28, 1)


# In[72]:


np.argmax(model.predict(image_data))


# In[73]:


plt.imshow(np.reshape(image_data, (28,28)), cmap='gray')
plt.title("Predicted Label %i" %np.argmax(model.predict(image_data)), fontsize = 15)
plt.show()


# In[ ]:




