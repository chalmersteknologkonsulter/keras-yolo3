from PIL import Image, ImageDraw, ImageFilter
from random import randint
from blurEdges import blurImage
from yolo3.utils import get_random_data
import csv
import numpy as np
import os
import glob

from matplotlib import pyplot as plt

def cutNGetImg(annotation_line,prefix):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(prefix + line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    image = np.array(image)/np.max(np.array(image))
    image_data = list()
    class_list = box[:, 4]
    box=box[:, :4]
    if (len(image.shape)==2):
        for b in range(len(box)):
            box1 = box[b, :]
            image_data.append(image[box1[1]:box1[3], box1[0]:box1[2]])
    else:
        for b in range(len(box)):
            box1 = box[b,:]
            image_data.append(image[box1[1]:box1[3], box1[0]:box1[2], :])

    return image_data, class_list

#TODO: Choose nr of images, blur images, define cooridnates and define what to write to annotation if num_sign=0

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'

road_path = 'C:\\Users\\CTK_CAD\\PycharmProjects\\BirdDetection\\keras-yolo3\\test_background\\' # background
#road_folder = os.listdir(road_path)  # create a folder to iterate through


aug_path = prefix + 'test/' # Final destination

'''
files = glob.glob(aug_path+'*.png')
for f in files:
    os.remove(f)

'''

road_size = (1200,900)
start = 501
max = start + 100 #number of images
number_of_roads = 20
num_classes = 3

# plt.bar(range(len(dist)), dist, 1/1.5, color="blue")
# plt.show()      #Plot distribution

for i in range(start,max):
    text_file = open(aug_path + "test_annotations.txt", "a")
    road_num = randint(1,number_of_roads) #Random road 'road_num.png'
    road = Image.open(road_path+str(road_num)+'.jpg')
    road = road.resize(road_size)
    num_of_signs_in_img = randint(1,3) #number of signs to be pasted into the image
    box_str = ''
    for j in range(0, num_of_signs_in_img):
        class_nr = np.random.choice(num_classes)  #chooses a class
        annotations_path = prefix + str(class_nr) + '/' + 'annotations' + str(class_nr) + '.txt'
        '''
        gtFile = open(annotations_path)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        '''
        #row_num = randint(1, sum(1 for row in gtReader))

        with open(annotations_path) as f:  # annotations file
            gtReader = f.readlines()  # read the txt annotations file

        a = np.random.choice(len(gtReader))

        sign, class_list = cutNGetImg(gtReader[a],prefix)



            #print(sign)
        for c, img in enumerate(sign):
            #print(img)

            if road_size[0]>img.shape[0] and road_size[1]>img.shape[1]:
                x = randint(0, road_size[0]-img.shape[1])
                y = randint(0, road_size[1]-img.shape[0])
                img = Image.fromarray(np.uint8(img*255))
                road = blurImage(road,img,x,y)
                box_str += ' ' + str(x)+','+str(y)+','+str(x+img.size[0])+','+str(y+img.size[1])+','+str(class_list[c])

    print('i:'+ str(i))
    text_file.write('augmented/'+format(i,'05d')+'.png'+box_str+'\n') #00600.png;774;411;815;446;11 in .txt-file, separate rows.


    road.save(aug_path + format(i, '05d')+'.png')
    text_file.close()


