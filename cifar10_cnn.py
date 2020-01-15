'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from PIL import Image
import csv

batch_size = 32
num_classes = 4
epochs = 75
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

load_train_images = False
try:
    x_train = np.load('train_images.npy')
    y_train = np.load('train_labels.npy')
except IOError:
    load_train_images = True
    print('Could not load train data from numpy files. Will load from image files instead')

load_test_images = False
try:
    x_test = np.load('test_images.npy')
    y_test = np.zeros((x_test.shape[0], 1), dtype=np.intc)
except IOError:
    load_test_images = True
    print('Could not load test data from numpy file. Will load from image files instead')

if load_train_images:
    with open('train.truth.csv') as train_csv_file:
        csv_reader = csv.reader(train_csv_file, delimiter=',')
        line_count = 0
        image_train_dict = {}
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                image_train_dict[row[0]] = row[1]
    train_csv_file.close()

    x_train = np.empty((len(image_train_dict), 64, 64, 3), dtype=np.ubyte)
    y_train = np.zeros((len(image_train_dict), 1), dtype=np.intc)

    for train_counter, image_train_name in enumerate(image_train_dict):
        image = Image.open('train/'+image_train_name)
        try:
            x_train[train_counter] = np.array(image, dtype=np.ubyte)
        except:
            print('Could not store', image_train_name, 'into numpy array')

        try:
            if image_train_dict[image_train_name] == 'rotated_right':
                y_train[train_counter] = 1
            elif image_train_dict[image_train_name] == 'upside_down':
                y_train[train_counter] = 2
            elif image_train_dict[image_train_name] == 'rotated_left':
                y_train[train_counter] = 3
            else:
                y_train[train_counter] = 0
        except:
            print('Could not get image orientation and store in array')

    numpy_file = open('train_images.npy','wb')
    np.save(numpy_file, x_train)
    numpy_file.close()
    numpy_file = open('train_labels.npy','wb')
    np.save(numpy_file, y_train)
    numpy_file.close()

if load_test_images:
    image_test_list = os.listdir('test')
    x_test = np.empty((len(image_test_list), 64, 64, 3), dtype=np.ubyte)
    y_test = np.zeros((len(image_test_list), 1), dtype=np.intc)

    for test_counter, image_test_name in enumerate(image_test_list):
        image = Image.open('test/'+image_test_name)
        try:
            x_test[test_counter] = np.array(image, dtype=np.ubyte)
        except:
            print('Could not store', image_test_name, 'into numpy array')

    numpy_file = open('test_images.npy', 'wb')
    np.save(numpy_file, x_test)
    numpy_file.close()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape, 'of type', y_train.dtype)
print('y_test shape:', y_test.shape, 'of type', y_test.dtype)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
