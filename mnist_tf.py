#!/usr/bin/env python
# coding: utf-8

# In[1]:


# initial data processing is from - https://medium.com/analytics-vidhya/mnist-digits-classification-with-deep-learning-using-python-and-numpy-4b33f0e1da32
#furthur analysis is from - https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# loading of data

# In[2]:



(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()


# In[3]:


#first reshape Y from (60000,) TO (60000,1)
#using to categorical convert to ONE HOT ENCODING
#finally transpose it
Y_tr_resh = Y_train_orig.reshape(60000, 1)
Y_te_resh = Y_test_orig.reshape(10000, 1)
Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
Y_te_T = to_categorical(Y_te_resh, num_classes=10)
Y_train = Y_tr_T
Y_test = Y_te_T


# In[4]:


#first flatten the data(convert 28*28 to a single 784 vector) and transpose it
#normalize it by dividing by 255
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1)
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.


# In[5]:


#define the relu and softmax output activation function
def relu(p):
    return np.maximum(0, p)
def softmax(u):
    return np.exp(u) / np.sum(np.exp(u), axis=0, keepdims=True)


# In[6]:


#initializations
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)
learning_rate = 0.005
n_iterations = 1000
batch_size = 128
dropout = 0.5


# In[7]:


#define the X and Y placeholders

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)


# In[8]:


#define weights and biases
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}


# In[9]:


#define the layer
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


# In[10]:


#define entropy and train using gradient descent
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[11]:


#define accuracy
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[12]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[13]:


#define next_batch function

epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]
    # for splitting out batches of data
def next_batch(batch_size):

    global X_train
    global Y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], Y_train[start:end]


# In[14]:


# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )


# In[15]:


#Test_data
test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)


# In[16]:


#to check the output
import numpy as np
from PIL import Image


# In[17]:


#we are downloading test image by typing the following code in anaconda promt and loading it and testing
#(curl -O https://raw.githubusercontent.com/do-community/tensorflow-digit-recognition/master/test_img.png)
img = np.invert(Image.open(r"C:\Users\Shravya\Downloads\0.png").convert('L')).ravel()


# In[18]:


img=img/255


# In[19]:


#the output should be 2
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))


# In[20]:


#to see the image
im=Image.open(r"C:\Users\Shravya\Downloads\0.png")
im.show()


# In[ ]:




