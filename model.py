import csv
import cv2
import numpy as np
import os
from math import ceil
from random import shuffle, seed

image_files = []
steering_angles = []
with open('../my_driving_data/driving_log_resamp.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num==1:
            image_files.append(line[1])
            steering_angles.append(line[2])

data = np.column_stack([image_files,steering_angles])
print(data.shape)
        

from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(data, test_size=0.2)

import sklearn
import tensorflow as tf
def generator(data, batch_size=32,train=False):
    seed(1)
    shuffle(data)
    num_samples = len(data)
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
                        if train:
                            rows,cols,depth = image.shape
                            angle = 5.0*np.random.normal()
                            mtx = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                            image = cv2.warpAffine(image,mtx,(cols,rows))
                            if np.random.uniform() > .5:
                                images.append(image)
                                steering_angles.append(float(entry[1]))
                            else:
                                images.append(cv2.flip(image,1))
                                steering_angles.append(-1*float(entry[1]))
                        else:
                            images.append(image)
                            steering_angles.append(float(entry[1]))
            
                X = np.asarray(images)
                y = np.array(steering_angles).reshape(len(X),1)
                yield sklearn.utils.shuffle(X, y)
        
if __name__ == "__main__":
    import sys
    
    batch_size = 32
    np.random.seed(50)
    train_generator = generator(train_data, batch_size=batch_size,train=True)
    validation_generator = generator(validation_data, batch_size=batch_size)
    
    if sys.argv[1] == "0":
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, BatchNormalization, Dropout, Activation

        model = Sequential()
        model.add(Cropping2D(input_shape=(160,320,3),cropping=((70,25),(0,0))))
        model.add(Lambda(lambda images: tf.map_fn(lambda input: tf.image.per_image_standardization(input),images)))

        model.add(Conv2D(filters=24,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=36,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=48,kernel_size=(5,5), strides=(2, 2)))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1, 1)))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(rate=.3))

        model.add(Flatten())

        model.add(Dense(100,activation='relu'))
        model.add(Dense(50,activation='relu'))
        model.add(Dense(10,activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mse',optimizer='adam')
        
    else:
        import h5py
        from keras.models import load_model
        model = load_model(sys.argv[1], custom_objects={'tf': tf})
        print('model loaded')
    
     
    from keras.callbacks import EarlyStopping
    model.fit_generator(train_generator,
                steps_per_epoch=ceil(len(train_data)/batch_size),\
                validation_data=validation_generator,\
                validation_steps=ceil(len(validation_data)/batch_size),\
                epochs=10, verbose=1, callbacks=[EarlyStopping(min_delta=0,patience=2)])
    
    save_choice = input("Do you want to save the model (y/n)? ")
    if save_choice == 'y':
        model.save('model.h5')
        print('Model saved as model.h5')
    del model