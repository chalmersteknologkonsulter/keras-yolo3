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
    def __init__(self, model_path, anchors_path, classes_path, score=0.2, iou=0.5, img_size=416):
        self.model_path = model_path  #'D:/Teade checkpoints/model_new_25_july_0_ver_1.h5' # only thing to change
        self.anchors_path = anchors_path  #'model_data/my_anchors4.txt' # only thing to change
        self.classes_path = classes_path  #'model_data/teade_classes.txt'
        self.score = score
        self.iou = iou
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (img_size, img_size) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

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
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,filename):
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

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        pred_prefix = 'C:/Users/CTK-VR1/PycharmProjects/mAP/predicted/'
        #pred_prefix = 'C:/Users/CTK_CAD/PycharmProjects/mAP/predicted/'
        text_file = open(pred_prefix + filename.split('.')[0] + '.txt', 'a')
        text_file.close()

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

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

def detect_img(yolo,prefix,anno_file):
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
    #yolo.close_session()

if __name__ == '__main__':
    prefix = 'C:/Users/CTK-VR1/Chalmers Teknologkonsulter AB/Bird Classification - Bird Detection - Images/Original images/'
    modelprefix = 'D:/Teade checkpoints/'
    annotations_file = 'test_anno.txt'  # File to evaluate mAP on
    anchors_path = 'model_data/my_anchors4.txt'
    classes_path = 'model_data/teade_classes.txt'
    score = 0.2
    IOU = 0.5
    model_path = 'model_old_23_july_170.h5'

    detect_img(YOLO(modelprefix + model_path, anchors_path, classes_path, score, IOU), prefix, annotations_file)  # predict using given model
