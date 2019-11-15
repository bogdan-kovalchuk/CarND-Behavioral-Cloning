import csv
import cv2
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Cropping2D


def generator(samples, batch_size=32, path_to_images='data/IMG/'):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # center img
                name = path_to_images + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # left img
                name = path_to_images + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                images.append(left_image)
                angles.append(center_angle + 0.2)

                # right img
                name = path_to_images + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                images.append(right_image)
                angles.append(center_angle - 0.2)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Conv2D layer
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

# Conv2D layer
model.add(Conv2D(32, 5, 5, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())

# Flattening
model.add(Flatten())

# Dense layer
model.add(Dense(32))
model.add(Dropout(0.20))

# Dense Layer
model.add(Dense(16))

# Dense Layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=5,
                    verbose=1)

model.save('model.h5')
