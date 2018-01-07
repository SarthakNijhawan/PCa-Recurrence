import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, concatenate, Activation,Dense, Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as keras
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
#from loss_function import loss_f, dice_f, metric_acc, dice_f_weighted_double
import math
import os
import cv2
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Global Variables
data_dump='gleason_data_dump' 
train_data_dir=data_dump+'/train'
val_data_dir=data_dump+'/valid'

img_rows = 299
img_cols = 299
nb_epoch =200
batch_size = 64

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# session = tf.Session(config=config)

sess = tf.Session()
keras.set_session(sess)

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.0001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
lrate = LearningRateScheduler(step_decay)

def small_net(img_rows, img_cols):

    inputs = Input((img_rows, img_cols, 3))
    
    conv1 = Conv2D(16, 6, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=7)(inputs)
    print("conv1 shape:",conv1.shape)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv1 = Conv2D(16, 5, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=6)(pool1)
    print("conv1 shape:",conv1.shape)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:",pool1.shape)

    conv2 = Conv2D(16, 4, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=4)(pool1)
    print("conv2 shape:",conv2.shape)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0)(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv2 = Conv2D(32, 3, padding = 'valid', kernel_initializer = 'he_normal', dilation_rate=2)(conv2)
    print("conv2 shape:",conv2.shape)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:",pool2.shape)
    
    conv3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv2 shape:",conv3.shape)
    # conv2 = BatchNormalization()(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool2 shape:",pool3.shape)
    
    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print("conv2 shape:",conv4.shape)
    # conv2 = BatchNormalization()(conv2)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print("pool2 shape:",pool4.shape)
    
    conv5 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print("conv2 shape:",conv5.shape)
    # conv2 = BatchNormalization()(conv2)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.8)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    print("pool2 shape:",pool4.shape)
    
    flatten = Flatten()(pool5)
    #fc_new1= Dense(512,activation='relu')(flatten)
    #drop_new1 = Dropout(0.9)(fc_new1)
    fc_new2 =Dense(64,activation='relu')(flatten)
    drop_new2=Dropout(0.8)(fc_new2)    
    
    fc_final_new =Dense(2,activation='softmax')(drop_new2)    
    model = Model(inputs = inputs, outputs = fc_final_new)
    
    return model

model = small_net(299,299)

model.compile(optimizer = Adam(lr=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
# model.load_weights("./weights/small_net_299_anno_cent_dilation_3.h5")

plot_model(model, to_file='model_plots/gleason_grade_small_net.png',show_shapes=True)

train_samples=len(os.listdir(train_data_dir+"/label_3/"))+len(os.listdir(train_data_dir+"/label_45/"))
val_samples=len(os.listdir(val_data_dir+"/label_3/"))+len(os.listdir(val_data_dir+"/label_45/"))

print("len train3:%d train45:%d",len(os.listdir(train_data_dir+"/label_3/")),len(os.listdir(train_data_dir+"/label_45/")))
print("len val3:%d train45:%d",len(os.listdir(val_data_dir+"/label_3/")),len(os.listdir(val_data_dir+"/label_45/")))
print("chance: ",1.0*len(os.listdir(val_data_dir+"/label_3/"))/val_samples)

class_weight = {0 : max(1.0,1.0*len(os.listdir(train_data_dir+"/label_45/"))/len(os.listdir(train_data_dir+"/label_3/"))),
    1: max(1.0,1.0*len(os.listdir(train_data_dir+"/label_3/"))/len(os.listdir(train_data_dir+"/label_45/")))}

checkpointer = ModelCheckpoint(filepath='./saved_models/gleason_299_anno_cent_dilation_4.h5', monitor='val_acc',verbose=1, save_best_only=True)
csv_logger = CSVLogger('./saved_models/gleason_299_anno_cent_dilation_4.csv')

def lab_color(input):
	return cv2.cvtColor(input,cv2.COLOR_BGR2Lab)

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(rescale=1.0/255.,preprocessing_function=None)#,samplewise_center=True,samplewise_std_normalization=True)
#datagen_augmented = ImageDataGenerator(rescale=1.0,horizontal_flip=True,vertical_flip=True,rotation_range=90,width_shift_range=0.2,height_shift_range=0.2,)#,preprocessing_function=images_aug)
datagen_augmented = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=90,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    channel_shift_range=0,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0/255.,
    preprocessing_function=None)

train_generator_augmented = datagen_augmented.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    # color_mode = 'grayscale',
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = datagen.flow_from_directory(
    val_data_dir,
    # color_mode = 'grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

print(model.summary())

#Fit the model on the batches generated by datagen.flow().
model.fit_generator(
    train_generator_augmented,
    steps_per_epoch=train_samples // batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=val_samples// batch_size,
    callbacks=[checkpointer,csv_logger,lrate],
    class_weight=class_weight)
