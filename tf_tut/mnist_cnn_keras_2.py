import tensorflow as tf
import numpy as np
import cv2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Model

from keras.datasets import mnist
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# @todo : local_response_normalization layer

# Keras layers can be called on TensorFlow tensors:

input_img = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(input_img)
conv1 = Activation('relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

conv2 = Conv2D(64, 5, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(conv1)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

conv2_flatten = Flatten()(conv2)
# conv2_flatten = tf.reshape(conv2, [-1, 7*7*64])

dense1 = Dense(1024, activation='relu')(conv2_flatten)
dense1 = Dropout(0.8)(dense1)

preds = Dense(10, activation='linear')(dense1)
preds = Activation('softmax')(preds)

model = Model(inputs=input_img, outputs=preds)

model.compile(optimizer='sgd',
	loss='categorical_crossentropy',
	metrics=['accuracy'])

print(model.summary())

model.fit(np.expand_dims(x_train, axis=4), y_train, epochs=1, verbose=1, batch_size=256)
model.save_weights("./tmp/models/mnist_cnn_keras.hdf5")

print(model.evaluate(np.expand_dims(x_test, axis=4), y_test))

model.load_weights("./tmp/models/mnist_cnn_keras.hdf5")
loaded_img = cv2.imread("img.jpg", 0)
loaded_img = np.reshape(loaded_img, (1, 28, 28, 1))
print(model.predict(loaded_img))
