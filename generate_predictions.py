import keras
import numpy as np
import sys, os

model = keras.models.load_model(sys.argv[1])

model_name, model_extension = os.path.splitext(os.path.basename(sys.argv[1]))

x_test = np.load('test_images.npy')
x_test = x_test.astype('float32')
x_test /= 255

predictions = model.predict(x_test, verbose=1)

print('Prediction size:', predictions.shape)

numpy_file = open('predictions/' + model_name + 'predictions.npy', 'wb')
np.save(numpy_file, predictions)
numpy_file.close()
