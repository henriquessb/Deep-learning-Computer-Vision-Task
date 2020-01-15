# Deep-learning Computer Vision Task
Deep-learning task for Deeper Systems job interview


I tried to accomplish the task of training and classification of rotated images using the CNN and ResNet Keras models for CIFAR10 dataset. The original source codes can be found in the links below.

https://keras.io/examples/cifar10_cnn/

https://keras.io/examples/cifar10_resnet/

Before I could run the codes, I had to install some python libraries (tensorflow, Keras, PIL) and other dependencies to run on gpu with better performance (CUDA 10.1, cuDNN 7.6.5.32). Everything was tested and ran on my personal system with Windows 10 Pro 64-bits using Python 3.7.4. My computer specs are i7 7820X, 32 GB DDR4 RAM and GTX 1080 Ti.

Using the models as a starting point. I successfully read the train and test image files, the labels on the csv file, converted the images and the labels to numpy arrays and used them for traning both models. I also created code to save the images and labels data converted to numpy arrays into files, so they can be read directly as numpy arrays if needed. The CNN model is in the file cifar10_cnn.py and the ResNet model is in the file cifar10_resnet.py.

To the moment of this writing, I could not get the trained data and the guessed orientations for the test dataset. But both models were showing arround 70% of accuracy for the train dataset and 25% for the test dataset (75 epochs for CNN and 20 epochs for ResNet). I also created a script (rotate_image.py) to read the test image files, generate the csv file with the image names and a rotation label (not from the traning, they are all rotated_left), generate the rotated images with .png format and the numpy array file of the rotated images. Even if I still don't have the trained data to correctly rotate the test images, I believe all required files are currently in this repository.


To run any of the models (CNN or ResNet) or the rotate_image script, the folders test/ and train/ with the image files and the train.truth.csv file are needed in the same directory as the python files. To run the CNN model, type on a console `python cifar10_cnn.py`. To run the ResNet model, type on a console `python cifar10_resnet.py`. To run the rotate_image script, type on a console `python rotate_image.py`.

