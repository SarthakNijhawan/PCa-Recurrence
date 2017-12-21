from __future__ import print_function
import os
import numpy as np
import scipy.stats as st
import tensorflow as tf
import cv2
import shutil
import time

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam,SGD, RMSprop
from keras import backend as K



def UniformKernel(ksize=101, sq_len=3):
	sq_len_shape = 2*sq_len + 1
	
	uniform2D = np.zeros((ksize, ksize))
	uniform2D[ksize/2-sq_len:ksize/2+sq_len+1, ksize/2-sq_len:ksize/2+sq_len+1] = \
		np.ones((sq_len_shape, sq_len_shape))
	return uniform2D

print(UniformKernel(21, 3))