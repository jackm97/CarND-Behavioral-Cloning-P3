import csv
import cv2
import numpy as np
<<<<<<< HEAD
import os
from math import ceil
from random import shuffle

data_dirs = os.listdir('../my_driving_data')

image_files = []
steering_angles = []
for data_dir in data_dirs:
    lines = []
    with open('../my_driving_data/' + data_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    for line in lines[1:]:
        centerfile = '../my_driving_data/' + data_dir + '/IMG/' + line[0].split('\\')[-1]
        image_files.append(centerfile)
        center_angle = float(line[3])
        steering_angles.append(str(center_angle))
        
        leftfile = '../my_driving_data/' + data_dir + '/IMG/' + line[1].split('\\')[-1]
        image_files.append(leftfile)
        left_angle = float(line[3]) - .2
        steering_angles.append(str(left_angle))
        
        rightfile = '../my_driving_data/' + data_dir + '/IMG/' + line[2].split('\\')[-1]
        image_files.append(rightfile)
        right_angle = float(line[3]) + .2
        steering_angles.append(str(right_angle))

data = np.array([image_files,steering_angles]).T

from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(data, test_size=0.2)

import sklearn
import tensorflow as tf
def generator(data, batch_size=32,train=False):
    num_samples = len(data)
    shuffle(data)
    image_input = tf.placeholder(tf.uint8,shape=(None,160,320,3))
    flipped = tf.reverse(image_input,axis=[2])
    with tf.Session() as sess:
        while 1:
            for offset in range(0,num_samples,batch_size):
                images = []
                steering_angles = []
                for entry in data[offset:offset+batch_size]:
                    if os.path.isfile(entry[0]):
                        image = cv2.imread(entry[0])
                        images.append(image)
                        steering_angles.append(float(entry[1]))
                X = np.asarray(images)
                y = np.array(steering_angles).reshape(len(X),1)
                if train:
                    flipped_images = sess.run(flipped,feed_dict={image_input:X})
                    np.concatenate((X,flipped_images))
                    np.concatenate((y,-1*y))
                yield sklearn.utils.shuffle(X, y)
        
if __name__ == "__main__":
    import sys
    
    batch_size = 32
    train_generator = generator(train_data, batch_size=batch_size,train=True)
    validation_generator = generator(validation_data, batch_size=batch_size)
    
    if sys.argv[1] == "0":
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, BatchNormalization, Dropout, Activation

        model = Sequential()
        model.add(Cropping2D(input_shape=(160,320,3),cropping=((70,25),(0,0))))
        model.add(Lambda(lambda images: tf.image.resize_images(images,size=(66,200))))
        model.add(Lambda(lambda images: tf.map_fn(lambda input: tf.image.per_image_standardization(input),images)))

        model.add(Conv2D(filters=24,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Lambda(lambda image: tf.nn.local_response_normalization(image)))

        model.add(Conv2D(filters=36,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Lambda(lambda image: tf.nn.local_response_normalization(image)))

        model.add(Conv2D(filters=48,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Lambda(lambda image: tf.nn.local_response_normalization(image)))

        model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Lambda(lambda image: tf.nn.local_response_normalization(image)))

        model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Lambda(lambda image: tf.nn.local_response_normalization(image)))

        model.add(Flatten())

        model.add(Dense(100,activation='relu'))
#         model.add(Dropout(rate=.3))
        model.add(Dense(50,activation='relu'))
#         model.add(Dropout(rate=.3))
        model.add(Dense(10,activation='relu'))
#         model.add(Dropout(rate=.3))
        model.add(Dense(1))

        model.compile(loss='mse',optimizer='adam')
        
    if sys.argv[1] == "1":
        import h5py
        from keras.models import load_model
        print("hi")
        model = load_model("model.h5", custom_objects={'tf': tf})
    
     
    from keras.callbacks import EarlyStopping
    model.fit_generator(train_generator,
                steps_per_epoch=ceil(len(train_data)/batch_size),\
                validation_data=validation_generator,\
                validation_steps=ceil(len(validation_data)/batch_size),\
                epochs=10, verbose=1, callbacks=[EarlyStopping(min_delta=.0001)])
    
    save_choice = input("Do you want to save the model (y/n)? ")
    if save_choice == 'y':
        model.save('model.h5')
        print('Model saved as model.h5')
    del model

=======

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines[1:]:
    image_path = './data/' + line[0]
    image = cv2.imread(image_path)
    try:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        images.append(image)
        measurement = np.float32(line[3])
        measurements.append(measurement)
    except:
        pass

images = np.array(images)
measurements = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse',metrics=['accuracy'],optimizer='adam',shuffle=True)

model.save('model.h5')
    
>>>>>>> 39456f14775ec6cd389158d22a1967b61f504af0
