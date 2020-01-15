# Deep-learning Computer Vision Task
Deep-learning task for Deeper Systems job interview


I tried to accomplish the task of training and classification of rotated images using the CNN and ResNet Keras models for CIFAR10 dataset. The original source codes can be found in the links below.

https://keras.io/examples/cifar10_cnn/

https://keras.io/examples/cifar10_resnet/

Before I could run the codes, I had to install some python libraries (tensorflow, Keras, PIL) and other dependencies to run on GPU with better performance (CUDA 10.1, cuDNN 7.6.5.32). Everything was tested and ran on my personal system with Windows 10 Pro 64-bits using Python 3.7.4. My computer specs are Core i7 7820X, 32 GB DDR4 RAM and GTX 1080 Ti.

Using the models as a starting point, I successfully read the train and test image files, the labels on the .csv file, converted the images and the labels to numpy arrays and used them for traning both models. I also created code to save the images and labels data converted to numpy arrays into files, so they can be read directly as numpy arrays if needed. The CNN model is in the file cifar10_cnn.py and the ResNet model is in the file cifar10_resnet.py. Also, the random flips of the images were disabled for the training. Beyond reading the data and training the networks, both scripts also save the model for further evaluation and prediction in the saved_models/ folder in files with .h5 extension. Another modification that was made to the training scripts was changing the number of classes (num_classes) to the amount of possible orientations (4 in this case).

After generating the model from the training, the generate_predictions script is used to predict the rotations of the images in the test data set. This script loads a model from a file with .h5 extension, executes the prediction on the test data and saves the prediction as a numpy array file in the predictions/ folder.

To the moment of this writing, the CNN model was showing 94% of accuracy for the train data set and the ResNet model 97%, but both models were showing around 25% for the test data set (50 epochs for CNN and 25 epochs for ResNet). The low score maybe is due to the numpy array for y_test being completely initialized with zeros. I also created a script (rotate_image.py) to read the test image files with corresponding rotation predictions in numpy array, generate the .csv file with the image names and corresponding rotation predictions, generate the rotated images with .png format and the numpy array file of the rotated images.


To run any of the models (CNN or ResNet) or the rotate_image script, the folders test/ and train/ with the image files and the train.truth.csv file are needed in the same directory as the python files. Also, the folders rotated/ and predictions/ folders must exist, even if they are empty. To run the CNN model, type on a console `python cifar10_cnn.py`. To run the ResNet model, type on a console `python cifar10_resnet.py`. To run the generate_predictions script, type on a console `python generate_predictions.py <name_of_the_model_file>.h5`. To run the rotate_image script, type on a console `python rotate_image.py <name_of_the_predictions_file>.npy`.
