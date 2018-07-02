from PIL import Image, ImageDraw, ImageFilter
from random import randint
from blurEdges import blurImage
from yolo3.utils import get_random_data
import csv
import numpy as np
import os
import glob

from matplotlib import pyplot as plt

def cutNGetImg(annotation_line):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    image = np.array(image)/np.max(np.array(image))
    image_data = list()
    box=box[:, :4]
    if (len(image.shape)==2):
        for b in range(len(box)):
            box1 = box[b, :]
            image_data.append(image[box1[1]:box1[3], box1[0]:box1[2]])
    else:
        for b in range(len(box)):
            box1 = box[b,:]
            image_data.append(image[box1[1]:box1[3], box1[0]:box1[2], :])

    return image_data, box

#TODO: Choose nr of images, blur images, define cooridnates and define what to write to annotation if num_sign=0

road_path = 'C:\\Users\\CTK_CAD\\PycharmProjects\\BirdDetection\\keras-yolo3\\background\\' # background
#road_folder = os.listdir(road_path)  # create a folder to iterate through

sign_path = 'C:\\Users\\CTK_CAD\\PycharmProjects\\BirdDetection\\keras-yolo3\\' #object
#sign_folder = os.listdir(sign_path)  # create a folder to iterate through

aug_path = 'C:\\Users\\CTK_CAD\\PycharmProjects\\BirdDetection\\keras-yolo3\\aug\\' # Final destination

files = glob.glob(aug_path+'*.png')
for f in files:
    os.remove(f)
files = glob.glob(aug_path+'\\boxes\\*.png')
for f in files:
    os.remove(f)  #Deletes all old augmented images

#00600.png-10000.png
#00600.png;774;411;815;446;11 in .txt-file, separate rows.
#00600.png;10;10;10;10;25
text_file = open("teade_annotations.txt", "w")
road_size = (800,500)
max = 100 #number of images
number_of_roads = 28
num_classes = 3

# plt.bar(range(len(dist)), dist, 1/1.5, color="blue")
# plt.show()      #Plot distribution

for i in range(max):
    road_num = randint(1,number_of_roads) #Random road 'road_num.png'
    road = Image.open(road_path+str(road_num)+'.jpg')
    road = road.resize(road_size)
    num_of_signs_in_img = randint(0,3) #number of signs to be pasted into the image
    box_str = ''
    for j in range(0,num_of_signs_in_img):
        class_nr = np.random.choice(num_classes)  #chooses a class
        prefix = 'C:\\Users\\CTK_CAD\\PycharmProjects\\BirdDetection\\keras-yolo3\\' + str(class_nr)
        annotations_path = prefix + '\\' + 'annotations' + str(class_nr) + '.txt'
        gtFile = open(annotations_path)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header

        row_num = randint(1, sum(1 for row in gtReader))
        with open(annotations_path) as f:  # annotations file
            gtReader = f.readlines()  # read the txt annotations file
        k=1
        for a in range(len(gtReader)):
            if k==row_num:
                sign, box = cutNGetImg(gtReader[a])
                #sign = Image.open(prefix + '\\' + row[0])
                #x1 = int(row[3])
                #y1 = int(row[4])
                #x2 = int(row[5])
                #y2= int(row[6])
                break;
            k+=1
            #print(sign)
        for img in sign:
            #print(img)
            x = randint(0, road_size[0]-img.shape[0])
            y = randint(0, road_size[1]-img.shape[1])
            img = Image.fromarray(np.uint8(img*255))
            road = blurImage(road,img,x,y)

        for b in range(len(box)):
            box1 = box[b, :]
            box_str += str(x+box1[0])+','+str(y+box1[1])+','+str(x+box1[2])+','+str(y+box1[3])+','+str(class_nr)+' '
    text_file.write('augmented/'+format(i,'05d')+'.png; '+box_str+'\n') #00600.png;774;411;815;446;11 in .txt-file, separate rows.

    #if(num_of_signs_in_img==0):
        #Vad ska skrivas till annotations om ingen skylt?
    road.save(aug_path + format(i, '05d')+'.png')
text_file.close()

'''
#Painting boxes
img_before = None
with open("teade_annotations.txt") as f:
    content = f.readlines()
for i in range(0,len(content)):  #loopa över alla rader i annotations
    semicolon = np.empty(6)
    d=0
    for j in range(0,len(content[i])):
        if(content[i][j] == ";" or content[i][j] == "\n"):
            semicolon[d] = j
            d = d+1
    x1 = int(content[i][int(semicolon[0])+1:int(semicolon[1])])  #koordinater för bounding box
    y1 = int(content[i][int(semicolon[1])+1:int(semicolon[2])])
    x2 = int(content[i][int(semicolon[2])+1:int(semicolon[3])])
    y2 = int(content[i][int(semicolon[3])+1:int(semicolon[4])])

    img_path_file = content[i][:int(semicolon[0])]    #filnamn
    if(img_before==None):
        img = Image.open(aug_path+img_path_file)
    else:
        if(img_path_file!=img_before):
            img.save(aug_path+'\\boxes\\' + img_before)
            img = Image.open(aug_path+img_path_file)
    draw = ImageDraw.Draw(img)
    draw.rectangle(((x1, y1), (x2, y2)),outline='red')
    draw.text((x1, y2), 'Class: '+content[i][int(semicolon[4])+1:int(semicolon[5])],(255,255,255)) #font=ImageFont.truetype("sans-serif.ttf", 16))
    del draw
    #img.show()
    #img.save(aug_path+content[i][:int(semicolon[0])]'.png')
    if(i==len(content)-1):
        img.save(aug_path + '\\boxes\\' + img_path_file)
    img_before = img_path_file '''
