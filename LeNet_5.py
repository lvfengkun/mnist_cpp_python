import os
import numpy as np
import keras
import pprint
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
#import matplotlib.pyplot as plt

#get_ipython().magic('matplotlib inline')

# # 加载手写字体
#
# 从keras自带的数据库中加载mnist手写字体

# In[154]:


img_rows, img_cols = (28, 28)
num_classes = 10


def get_mnist_data():
    """
    加载mnist手写字体数据集
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_mnist_data()
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# # LetNet5 网络模型架构

# In[155]:


def LeNet5(w_path=None):
    input_shape = (img_rows, img_cols,1)
    img_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = Conv2D(64, (3, 3), activation="relu", padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Dropout(0.25)(x)

    x = Flatten(name='flatten')(x)

    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='LeNet5')
    if (w_path): model.load_weights(w_path)

    return model


lenet5 = LeNet5()
print('Model loaded.')
lenet5.summary()

lenet5.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
lenet5.fit(x_train, y_train,
         batch_size = 128,
         epochs = 1,
         verbose = 1,
         validation_data = (x_test, y_test),
        )