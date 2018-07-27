
import keras.backend as K
from keras.layers import Input, Lambda
import numpy as np
from PIL import Image
import os

'''
Change annotation_path for custom data
'''
def prep(prefix,filename):
    print('Extracting Test data')
    #prefix = 'C:/Users/CTK-VR1/Chalmers Teknologkonsulter AB/Bird Classification - Bird Detection - Images/Original images/'
    # prefix = 'gs:/bdp-original-images/Original images/'
    #annotation_path = prefix + 'test3.txt'
    annotation_path = prefix + filename

    classes = ["Drone", "Bird", "Airplane"]
    to_img_path = 'C:/Users/CTK-VR1/PycharmProjects/mAP/images/'

    to_gt_path = 'C:/Users/CTK-VR1/PycharmProjects/mAP/ground-truth/'

    for subdir, dirs, files in os.walk(to_gt_path):
        for f in files:
            os.remove(os.path.join(subdir, f))

    for subdir, dirs, files in os.walk(to_img_path):
        for f in files:
            os.remove(os.path.join(subdir, f))

    with open(annotation_path) as f:
        reader = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(reader)

    nr_of_lines = len(reader)

    train_frac = 0
    #train_frac = 0.9

    train_num = int(nr_of_lines*train_frac)

    test_lines = reader[train_num:]


    for line in test_lines:
        full_test_file = open(prefix + 'test_data.txt', "a")
        full_test_file.write(line)
        full_test_file.close()
        name = line.split()[0]
        boxes = line.split()[1:]
        test_img = Image.open(prefix + name)
        test_img.save(to_img_path + name)
        text_file = open(to_gt_path + name.split('.')[0] + '.txt', "a")
        for box in boxes:
            coord = box.split(',')[0:4]
            class_nr = box.split(',')[4]
            class_name = classes[int(class_nr)]
            text_file.write(class_name +' '+ coord[0]+' '+ coord[1]+' '+ coord[2]+' '+ coord[3]+'\n')
        text_file.close()

    print("Test data extracted")