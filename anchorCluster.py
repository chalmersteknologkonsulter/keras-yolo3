import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'

annotationfile = prefix + 'newAnn.txt'
anchorsfile = 'my_anchors.txt'

n_kluster=9
ratios = []

with open(annotationfile) as f:
    lines = f.readlines()

for line in lines[:len(lines)-500]:
    line = line.strip('\n').split(' ')
    filename = line[0]
    area_factor = 1
    #if True:   #Scale all images
    #if filename[:5] == 'teade':
    #    img = Image.open(prefix+filename)
    #    area_factor = img.size[0]*img.size[1]/(256*256)

    boxes = line[1:]
    for box in boxes:
        box = box.split(',')
        width = (int(box[2])-int(box[0]))/np.sqrt(area_factor)
        height = (int(box[3])-int(box[1]))/np.sqrt(area_factor)
        ratios.append([width,height])

ratios = np.array(ratios)

km = KMeans(n_clusters = n_kluster)
pred = km.fit_predict(ratios)

clusters= list()

xyMean = np.zeros((n_kluster,2))

anchFile = open(anchorsfile,'a')

for i in range(n_kluster):
    idx = pred == i
    hej = ratios[idx, :]
    xMean, yMean = np.mean(hej, axis=0)
    xyMean[i,:]=(xMean,yMean)
    clusters.append(hej)
    plt.scatter(hej[:, 0], hej[:, 1])
   # plt.show()


plt.scatter(ratios[:,0],ratios[:,1], c = pred)
plt.scatter(xyMean[:,0],xyMean[:,1],c="red")
plt.show()

xyMean.sort(axis=0)

for c,mean in enumerate(xyMean):
    if c<n_kluster-1:
        anchFile.write(str(int(mean[0] + 0.5)) + ',' + str(int(mean[1] + 0.5)) + ',\t')
    else:
        anchFile.write(str(int(mean[0] + 0.5)) + ',' + str(int(mean[1] + 0.5)))










