import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Dropout, Cropping2D


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


# Read csv and populate samples
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Set batch size
batch_size = 16

# Split data to train and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(160, 320, 3)))

# Crop region of interest
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Conv2D layer with filters 16, kernel size 5x5, strides 2x2 and 'relu' activation function
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))

# Conv2D layer with filters 24, kernel size 5x5, strides 2x2 and 'relu' activation function
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation="relu"))

# Conv2D layer with filters 32, kernel size 5x5, strides 2x2 and 'relu' activation function
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))

# Conv2D layer with filters 48, kernel size 3x3, strides 1x1 and 'relu' activation function
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))

# Conv2D layer with filters 64, kernel size 3x3, strides 1x1 and 'relu' activation function
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))

# Flattens the input
model.add(Flatten())

# Dense layer of size 90
model.add(Dense(90))
model.add(Dropout(0.05))

# Dense layer of size 50
model.add(Dense(50))
model.add(Dropout(0.05))

# Dense layer of size 10
model.add(Dense(10))
model.add(Dropout(0.05))

# Dense layer of size 1
model.add(Dense(1))

# Compile and train model with 'adam' optimizer and 'mse' loss function
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=np.ceil(len(validation_samples)/batch_size),
                                     epochs=5,
                                     verbose=1)

# Save results to 'hdf5' file to future usage
model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
