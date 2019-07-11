import csv
import cv2
import numpy as np

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
        print(image.shape)
    except:
        pass
#     if !image.empty():
#         image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#         images.append(image)
#         measurement = np.float32(line[3])
#         measurements.append(measurement)

# images = np.array(images)
# measurements = np.array(measurements)

# from keras.models import Sequential
# from keras.layers import Flatten, Dense

# model = Sequential()
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))

# model.compile(loss='mse',optimize='adam',shuffle=True)

# model.save('model.h5')
    
