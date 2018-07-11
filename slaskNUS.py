from PIL import Image
import numpy as np
'''
img = Image.open('E:/Downloads/NUS_small.jpg')
img.show()
print(img.size)
#img = img.resize((453,340))
img = img.resize((340,453))
print(img.size)
img.show()
img.save('E:/Downloads/NUS_small2.jpg')
x = 340/453
print(str(x))

'''

'''
prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
annotationsfile = prefix + 'annotations_val.txt'

with open(annotationsfile) as f:
    lines = f.readlines()


for line in lines:
    boxes = line.split()[1:]
    name = line.split()[0]
    new_boxes = []
    data = ''
    for box in boxes:
        left, top, w, h, c = box.split(',')
        right = int(left) + int(w)
        bottom = int(top) + int(h)
        if data == '':
            data += name + ' ' + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bottom) + ',' + str(c)
        else:
            data += ' ' + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bottom) + ',' + str(c)

    data += '\n'
    # print('data: '+data)
    file = open(prefix + 'annotations_val_formatted.txt', 'a')
    file.write(data)
    file.close()
    
'''
'''
prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
annotationsfile = prefix + 'train.txt'

with open(annotationsfile) as f:
    lines = f.readlines()
for line in lines:
    if line[-2]==' ':
        line=line[:-2]+'\n'

    file = open(prefix + 'train2.txt', 'a')
    file.write(line)
    file.close()
'''
'''
from argparse import ArgumentParser as ap

parser = ap(description='Radius/height container')
parser.add_argument('radius',type=int, help='Radius of t')
parser.add_argument('height',type=int, help='h of t')
args = parser.parse_args()


def cyl_vol(r,h):
    vol = h*r
    return vol
if __name__== '__main__':

    print(cyl_vol(r,h))


'''
for i in range(250,252):
    print(i)