import numpy as np

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
annotationsfile = prefix + 'test_drones.txt'

class_counter = np.zeros((1,3))


with open(annotationsfile) as f:
    lines = f.readlines()
for line in lines:
    boxes = line.split()[1:]
    for box in boxes:
        l,t,r,b,c = box.split(',')
        class_counter[0,int(c)] += 1


for i in range(3):
    print(class_counter[0,i])