from PIL import Image, ImageDraw

prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
annotationsfile = prefix + 'test_anno.txt'
#annotationsfile = prefix + 'newAnn.txt'
imagepath = 'C:/Users/CTK_CAD/PycharmProjects/mAP/images/'

with open(annotationsfile) as f:
    lines = f.readlines()

for line in lines:
    line = line.strip('\n')
    linesplits = line.split()
    filename = linesplits[0]
    img = Image.open(imagepath+filename)
    #img.show()
    boxes = linesplits[1:]
    for box in boxes:
        boxParts = box.split(',')
        x1 = int(boxParts[0])
        y1 = int(boxParts[1])
        x2 = int(boxParts[2])
        y2 = int(boxParts[3])
        #print([x1,x2,y1,y2])
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1,y1,x2,y2])
    #img.show()
