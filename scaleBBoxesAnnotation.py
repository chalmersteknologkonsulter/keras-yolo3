import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original, rescaled 256x256/'

annotationfile = prefix + 'train.txt'

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
outfile = 'annotations2.txt'

with open(annotationfile) as f:
    lines = f.readlines()

fileout = open(outfile, 'a')

for line in lines:
    temp = line
    line = line.strip('\n').split(' ')
    filename = line[0]
    scale_factor = (1,1)
    if filename[:5] != 'teade':
        img = Image.open(prefix+filename)
        scale_factor = (img.size[0]/256, img.size[1]/256)
        boxes = line[1:]
        fileout.write(filename)
        for box in boxes:
            box = box.split(',')
            x1 = int(int(box[0]) * scale_factor[0] + 0.5)
            y1 = int(int(box[1]) * scale_factor[1] + 0.5)
            x2 = int(int(box[2]) * scale_factor[0] + 0.5)
            y2 = int(int(box[3]) * scale_factor[1] + 0.5)
            fileout.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+box[4])
        fileout.write('\n')
    else:
        fileout.write(temp)

fileout.close()