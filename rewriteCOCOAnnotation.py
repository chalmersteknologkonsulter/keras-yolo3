from PIL import Image
import numpy as np
import io
from matplotlib import pyplot as plt
import os

annotationsfile = 'annotations_train.txt'
outfile = 'annotations_train2.txt'

with open(annotationsfile) as f:
    lines = f.readlines()

fout = open(outfile,'a')

for line in lines:
    line = line.strip('\n')
    linesplits = line.split(' ')
    filename = linesplits[0]
    fout.write(filename)
    boxes = linesplits[1:]
    for box in boxes:
        boxParts = box.split(',')
        x2 = int(boxParts[0])+int(boxParts[2])
        y2 = int(boxParts[1])+int(boxParts[3])
        box = ' '+str(boxParts[0])+','+str(boxParts[1])+','+str(x2)+','+str(y2)+','+boxParts[4]
        fout.write(box)
    fout.write('\n')
fout.close()
