import os, sys
import csv
import numpy as np
from PIL import Image

with open('test.preds.csv', 'w') as output_csv_file:
    image_list = os.listdir('test')
    output_csv_file.write('fn,label\n')
    y_train = np.load(sys.argv[1])
    for i, aImage in enumerate(image_list):
        string_output = aImage
        if np.argmax(y_train[i]) == 1:
            string_output += ',rotated_right'
        elif np.argmax(y_train[i]) == 2:
            string_output += ',upside_down'
        elif np.argmax(y_train[i]) == 3:
            string_output += ',rotated_left'
        else:
            string_output += ',upright'
        output_csv_file.write(string_output+'\n')
output_csv_file.close()

with open('test.preds.csv') as input_csv_file:
    csv_reader = csv.reader(input_csv_file, delimiter=',')
    line_count = 0
    image_dict = {}
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            image_dict[row[0]] = row[1]
input_csv_file.close()

image_success_counter = 0
image_array = np.empty((5361, 64, 64, 3), dtype='uint8')

for image_counter, aImage in enumerate(image_dict):
    file_name = aImage
    image_name, image_extension = os.path.splitext(file_name)
    im = Image.open('test/' + file_name)

    orientation = image_dict[aImage]
    if orientation == 'rotated_right':
        im = im.rotate(90)
    elif orientation == 'upside_down':
        im = im.rotate(180)
    elif orientation == 'rotated_left':
        im = im.rotate(270)

    try:
        im.save('rotated/' + image_name + '.png')
        image_success_counter += 1
    except IOError:
        print('Could not save', image_name, 'as .png file')

    try:
        image_array[image_counter] = np.array(im, dtype='uint8')
    except:
        print('Could not store', image_name, 'into numpy array')

    im.close()

output_numpy_file = open('rotated_test_images.npy', 'wb')
np.save(output_numpy_file, image_array)
output_numpy_file.close()

print('Expected', line_count-1, 'image files from csv file and', image_success_counter, 'were successfully opened and rotated')

# generate the network including the training with epochs and the feedfowrard
# make the network train for the orientation and output the guessed orientation
# feed the network with training images and ground truth csv
# apply the trained network into test data and save guessed orientation into csv
