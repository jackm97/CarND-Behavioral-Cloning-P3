import csv
import cv2
import numpy as np
import os
from math import ceil
from random import shuffle, seed

# Gather image filenames and associated steering angles
image_files = []
steering_angles = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not reader.line_num==1:
            image_files.append(line[0])
            steering_angles.append(line[3])

data = np.column_stack([image_files,steering_angles])
        
# shuffle and split data
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(data, test_size=0.2)

# generator for data to reduce memory load 
import sklearn
import tensorflow as tf
def generator(data, batch_size=32,train=False):
    seed(1)
    shuffle(data)
    num_samples = np.shape(data)
    num_samples = num_samples[0]
    while 1:
        for offset in range(0,num_samples,batch_size):
            images = []
            steering_angles = []
            for entry in data[offset:offset+batch_size]:
                if os.path.isfile("./data/data/" + entry[0].strip()):
                    image = cv2.imread("./data/data/" + entry[0].strip())
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # randomly rotate and flip image
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
            X, y = sklearn.utils.shuffle(X, y)
            yield (X, y)
        
if __name__ == "__main__":
    import sys
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    batch_size = 100
    np.random.seed(50)
    train_generator = generator(train_data, batch_size=batch_size,train=True)
    validation_generator = generator(validation_data, batch_size=batch_size)
    
    # create model and train from scratch
    if sys.argv[1] == "0":
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, BatchNormalization, Dropout, Activation

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
    
    # train pretrained model
    else:
        import h5py
        from keras.models import load_model
        model = load_model(sys.argv[1], custom_objects={'tf': tf})
        print('model loaded')
    
    # training 
    from keras.callbacks import EarlyStopping
    model.fit_generator(train_generator,
                steps_per_epoch=ceil(len(train_data)/batch_size),\
                validation_data=validation_generator,\
                validation_steps=ceil(len(validation_data)/batch_size),\
                epochs=50, verbose=1, callbacks=[EarlyStopping(min_delta=0,patience=5)])
    
    # save model
    save_choice = input("Do you want to save the model (y/n)? ")
    if save_choice == 'y':
        model.save('model.h5')
        print('Model saved as model.h5')
    del model
