#!/usr/bin/env python
# coding: utf-8

# In[30]:


##import tensoflow
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import optimizers


# In[31]:


## the data is available directly in the tf.keras datasets API we can load it lke this 
mnist = tf.keras.datasets.fashion_mnist


# In[32]:


### training and testing data 
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# In[33]:


## !!! we can change the number of label and show the result
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[1])
print(training_labels[1])
print(training_images[1])


# In[34]:


training_images  = training_images / 255.0
test_images = test_images / 255.0


# In[35]:


## now we have to design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[36]:



model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)


# In[37]:


model.evaluate(test_images, test_labels)


# In[ ]:





# In[25]:


classifications = model.predict(test_images)

print(classifications[0])


# In[26]:


print(test_labels[0])

