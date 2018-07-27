import numpy as np

class_counter = np.zeros((1,3))

prefix = 'C:/Users/CTK-VR1/Chalmers Teknologkonsulter AB/Bird Classification - Bird Detection - Images/Original images/'
#prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
#prefix = 'gs:/bdp-original-images/Original images/'
annotation_path = prefix + 'train2.txt'

val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)

train_lines = lines[:-num_val]
val_lines = lines[-num_val:]
out_train = open('train3.txt','a')

for line in train_lines:
    out_train.write(line)


out_train.close()

out_val = open('new_test.txt','a')
for line in val_lines:
    out_val.write(line)

out_val.close()

for line in lines:
    boxes = line.split()[1:]
    for box in boxes:
        l,t,r,b,c = box.split(',')
        class_counter[0,int(c)] += 1

for i in range(3):
    print(class_counter[0,i])

########
