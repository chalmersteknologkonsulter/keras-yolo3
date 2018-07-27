from prepare_test_set import prep
from predict import YOLO, detect_img
from mAP.calculate_mAP import calcAndSavemAP
from matplotlib import pyplot as plt
import os

##########Choose models to test and test set

model_paths = []

if os.path.exists('D:/Teade checkpoints/model_names_index'):  # if it exist already
    # reset the results directory
    os.remove('D:/Teade checkpoints/model_names_index.txt')
    print('removed')
for subdir, dirs, files in os.walk('D:/Teade checkpoints/'):
    for f in files:
        #outfile = open('D:/Teade checkpoints/model_names_index.txt','a')
        file = os.path.join(subdir, f)
        #outfile.write(file+'\n')
        #outfile.close()
        model_paths.append(file)

#model_base = 'model_new_26_july_30_ver'
#model_versions = [1, 2]
prefix = 'C:/Users/CTK-VR1/Chalmers Teknologkonsulter AB/Bird Classification - Bird Detection - Images/Original images/'
annotations_file = 'test_anno.txt'  # File to evaluate mAP on
#prep(prefix, annotations_file)  # prepares files to be evaluated on, only run if you change annotations_file
############### Below: Very rarely touch

anchors_path = 'model_data/my_anchors4.txt'
classes_path = 'model_data/teade_classes.txt'

for score in [0.2]:
    IOU = 0.5

    xaxis = 'Evaluation on all models (score {}, IOU {}'.format(score, IOU) #hat does the x-axis signify in the graph, i.e. layers frozen

    result_prefix = 'C:/Users/CTK-VR1/PycharmProjects/mAP/results/final/'
    result_prefix += annotations_file.split('.txt')[0] + '/{}/'.format(score)

    mAP = []
    ap_bird = []
    ap_airplane = []
    ap_drone = []

    if False: # Loops over different models
        for model_path in model_paths:
            model_name = model_path.split('/')[2].split('.h5')[0]
            print('Predicting on test set "{}" with model "{}"'.format(annotations_file,model_path))
            detect_img(YOLO(model_path, anchors_path, classes_path, score, IOU), prefix, annotations_file)  # predict using given model
            temp_ap_per_class, temp_mAP = calcAndSavemAP(result_prefix,model_name)  #ap_per_class: class_name, AP,
            mAP.append(temp_mAP)
            for clas in temp_ap_per_class:
                if clas[0] == 'Airplane':
                    ap_airplane.append(clas[1]*100)
                if clas[0] == 'Bird':
                    ap_bird.append(clas[1]*100)
                if clas[0] == 'Drone':
                    ap_drone.append(clas[1]*100)

    if True: #loops over different img sizes
        for i in range(12):
            model_name = 'model_old_23_july_170'+'_img_size_'+str(i*64+320)
            print('Predicting on test set "{}" with model "{}"'.format(annotations_file, model_name))
            detect_img(YOLO('D:/Teade checkpoints/model_old_23_july_170.h5', anchors_path, classes_path, score, IOU,320+i*64), prefix, annotations_file)  # predict using given model
            temp_ap_per_class, temp_mAP = calcAndSavemAP(result_prefix, model_name)  #ap_per_class: class_name, AP,
            mAP.append(temp_mAP)
            for clas in temp_ap_per_class:
                if clas[0] == 'Airplane':
                    ap_airplane.append(clas[1]*100)
                if clas[0] == 'Bird':
                    ap_bird.append(clas[1]*100)
                if clas[0] == 'Drone':
                    ap_drone.append(clas[1]*100)
    plt.plot(mAP, 'r-', label='mAP')
    plt.plot(ap_airplane, 'b--', label='AP Airplane')
    plt.plot(ap_drone, 'g--', label='AP Drone')
    plt.plot(ap_bird, 'm--', label='AP Bird')
    plt.xticks(rotation=90)
    plt.title('mAP for different models\nEvaluated on {}'.format(annotations_file.split('.txt')[0]))
    plt.xlabel(xaxis)
    plt.ylabel('mAP (%)')
    plt.legend()
    #plt.savefig(result_prefix+'mAP_test_anno_{}.png'.format(score))
    plt.savefig(result_prefix + 'mAP_test_anno_diff_size.png')
    #plt.show()