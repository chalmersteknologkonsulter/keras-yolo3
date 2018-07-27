#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import glob
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

'''
before run, run prepare_test_set.py
Only change model_path, anchors_path and score!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

class YOLO(object):
    def __init__(self):
        self.model_path = ['D:/Teade checkpoints/model_old_23_july_170.h5', 'D:/Teade checkpoints/model_old_23_july_200.h5'] # only thing to change
        self.anchors_path = 'model_data/my_anchors4.txt' # only thing to change
        self.classes_path = 'model_data/teade_classes.txt'
        self.score = 0.2
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.yolo_model = []
        self.generate()
        #self.boxes, self.scores, self.classes =

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        self.boxes, self.scores, self.classes, self.input_image_shape = [], [], [], []
        for i in range(len(self.model_path)):
            temp_path = os.path.expanduser(self.model_path[i])
            assert temp_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
            try:
                temp_model = load_model(temp_path, compile=False)
            except:
                temp_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                    if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
                temp_model.load_weights(temp_path) # make sure model, anchors and classes match
            else:
                assert temp_model.layers[-1].output_shape[-1] == \
                    num_anchors/len(temp_model.output) * (num_classes + 5), \
                    'Mismatch between model and given anchor and class sizes'
            self.yolo_model.append(temp_model)
            print('{} model, anchors, and classes loaded.'.format(temp_path))

            # Generate output tensor targets for filtered bounding boxes.
            self.input_image_shape.append(K.placeholder(shape=(2, )))
            boxes, scores, classes = yolo_eval(self.yolo_model[i].output, self.anchors,
                    len(self.class_names), self.input_image_shape[i],
                    score_threshold=self.score, iou_threshold=self.iou)
            self.boxes.append(boxes)
            self.scores.append(scores)
            self.classes.append(classes)
        #return boxes, scores, classes

    def detect_image(self, image, filename):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = [], [], []
        j=0
        for model in self.yolo_model:
            temp_out_boxes, temp_out_scores, temp_out_classes = self.sess.run(
                [self.boxes[j], self.scores[j], self.classes[j]],
                feed_dict={
                    model.input: image_data,
                    self.input_image_shape[j]: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            out_boxes.append(temp_out_boxes)
            out_scores.append(temp_out_scores)
            out_classes.append(temp_out_classes)
            j+=1
        final_out_boxes = list(out_boxes[0])
        final_out_classes = list(out_classes[0])
        final_out_scores = list(out_scores[0])

        IOU_thresh = 0.2

        model_idx = 1
        for model_boxes in out_boxes[1:]:
            for box_idx, box in enumerate(model_boxes):
                have_overlapped=False
                for current_idx, current_best_box in enumerate(final_out_boxes):
                    if my_IOU(box, current_best_box) > IOU_thresh and final_out_classes[current_idx] == out_classes[model_idx][box_idx]:
                        have_overlapped = True
                        if final_out_scores[current_idx] < out_scores[model_idx][box_idx]:
                            final_out_boxes[current_idx] = box
                            final_out_scores[current_idx] = out_scores[model_idx][box_idx]
                        break
                if not have_overlapped:
                    final_out_boxes.append(box)
                    final_out_classes.append(out_classes[model_idx][box_idx])
                    final_out_scores.append(out_scores[model_idx][box_idx])

            model_idx += 1

        #print('Found {} boxes for {}'.format(len(final_out_boxes), 'img'))

        pred_prefix = 'C:/Users/CTK-VR1/PycharmProjects/mAP/predicted/'
        #pred_prefix = 'C:/Users/CTK_CAD/PycharmProjects/mAP/predicted/'
        text_file = open(pred_prefix + filename.split('.')[0] + '.txt', 'a')
        text_file.close()

        for i, c in reversed(list(enumerate(final_out_classes))):
            predicted_class = self.class_names[c]
            box = final_out_boxes[i]
            score = final_out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            text_file = open(pred_prefix + filename.split('.')[0]+'.txt','a')
            text_file.write(label + ' ' +str(left)+' '+str(top)+' '+str(right)+' '+str(bottom) + '\n')
            text_file.close()

    def close_session(self):
        self.sess.close()

def my_IOU(box1, box2):
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2
    area1 = np.abs(bottom1-top1) * np.abs(left1-right1)
    area2 = np.abs(bottom2-top2) * np.abs(left2-right2)

    x_left = np.max((top1, top2))
    y_top = np.max((left1, left2))
    x_right = np.min((bottom1, bottom2))
    y_bottom = np.min((right1, right2))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    IOU_score = intersection_area / (area1 + area2 - intersection_area)
    return IOU_score


def detect_img(yolo):
    prefix = 'C:/Users/CTK-VR1/Chalmers Teknologkonsulter AB/Bird Classification - Bird Detection - Images/Original images/'
    #prefix = 'C:\\Users\\CTK_CAD\\Chalmers Teknologkonsulter AB\\Bird Classification - Images\\Bird Detection - Images\\Original images\\'
    anno_file = 'test_anno.txt'

    pred_dir = 'C:/Users/CTK-VR1/PycharmProjects/mAP/predicted'
    #pred_dir = 'C:/Users/CTK_CAD/PycharmProjects/mAP/predicted'
    for subdir, dirs, files in os.walk(pred_dir):
        for f in files:
            os.remove(os.path.join(subdir,f))

    with open(prefix+anno_file) as f:
        lines = f.readlines()

    for line in lines:
        filename = line.split()[0]
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = Image.open(prefix + filename)
            #print(filename)
            yolo.detect_image(image, filename)
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
