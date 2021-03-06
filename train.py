"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import my_metrics as metrics


from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

def map50_metric(ytrue, ypred):
    return metrics.map_metric(ytrue, ypred, 0.5, 3)





def _main():
    prefix = 'C:/Users/CTK_CAD/Chalmers Teknologkonsulter AB/Bird Classification - Images/Bird Detection - Images/Original images/'
    #prefix = 'gs:/bdp-original-images/Original images/'
    annotation_path = prefix + 'train2.txt'
    log_dir = 'E:/Teade checkpoints/000/'
    event_dir = 'logs/000/'
    classes_path = 'model_data/teade_classes.txt'
    anchors_path = 'model_data/my_anchors4.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (320,320) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=1, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        #model = create_model(input_shape, anchors, num_classes,freeze_body=2, weights_path='logs/000/new_trained_weights_stage_1.h5') # make sure you know what you freeze
        model = create_model(input_shape, anchors, num_classes,freeze_body=2, weights_path=log_dir + 'finalmodel_stage_17.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=event_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True,mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if False:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        #print(model.metrics_names)

        #print(model.summary())
        #print(model.layers[185].trainable)  #Layer 184 = last Add-layer, end of Darknet?  Freeze=1 freezes everythin up to this point

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], prefix, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], prefix, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=250,
                verbose=1,
                initial_epoch=180,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'finalmodel_stage_XXX.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.

    ################################# GPU limitations:
    '''
    Batch 16:
    Input_size = 352, batch 16 funkar ner t.o.m. 110 frysta lager
    Input_size = 320, batch 16 funkar ner t.o.m. 100 frysta lager
    Input_size = 288, batch 16 funkar ner t.o.m. 80 frysta lager
    Input_size = 256, batch 16 funkar ner t.o.m. 50 frysta lager
    Input_size = 224, batch 16 funkar ner t.o.m. 20 frysta lager
    Input_size = 192, batch 16 funkar ner t.o.m. 10 frysta lager
    Input_size = 160, batch 16 funkar ner t.o.m. 0 frysta lager - hela modellen!
    
    Batch 12:
    Input_size = 352, batch 12 funkar ner t.o.m. 90 frysta lager
    Input_size = 320, batch 12 funkar ner t.o.m. 70 frysta lager
    Input_size = 288, batch 12 funkar ner t.o.m. 50 frysta lager
    Input_size = 256, batch 12 funkar ner t.o.m. 20 frysta lager
    Input_size = 224, batch 12 funkar ner t.o.m. 10 frysta lager
    Input_size = 192, batch 12 funkar ner t.o.m. 0 frysta lager - hela modellen!
    Input_size = 160, batch 12 funkar ner t.o.m. 0 frysta lager - hela modellen!
    
    Batch 8:
    Input_size = 352, batch 8 funkar ner t.o.m. 40 frysta lager
    Input_size = 320, batch 8 funkar ner t.o.m. 30 frysta lager
    Input_size = 288, batch 8 funkar ner t.o.m. 20 frysta lager
    Input_size = 256, batch 8 funkar ner t.o.m. 10 frysta lager
    Input_size = 224, batch 8 funkar ner t.o.m. 0 frysta lager - hela modellen!
    Input_size = 192, batch 8 funkar ner t.o.m. 0 frysta lager - hela modellen!
    Input_size = 160, batch 8 funkar ner t.o.m. 0 frysta lager - hela modellen!
    
    Batch 6:
    Input_size = 320, batch 6 funkar ner t.o.m. 10 frysta lager
    
    Batch 4:
    Input_size = 320, batch 4 funkar ner t.o.m. 0 frysta lager - hela modellen!
    
    '''

    ####################################

    num_iter = 18
    epoch_per_iter = 15
    start_freeze = 170
    freeze_per_iter = 10
    start_epoch = 330
    start_model = 18 #save as Final_model_XXX.h5 for the first iteration
    batch_size = 16
    if True:
        for iter in range(num_iter):
            temp = start_freeze-iter*freeze_per_iter
            for i in range(temp,len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Freeze '+ str(temp) +' of the layers.')

            if temp <= 110 and temp >=90:  #antar input_sixe = 320
                batch_size = 12
            if temp <= 80 and temp >=50:
                batch_size = 8
            if temp<=40:
                batch_size=4

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            init_epoch = start_epoch+iter*epoch_per_iter
            model.fit_generator(data_generator_wrapper(lines[:num_train], prefix, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], prefix, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=init_epoch+epoch_per_iter,
                initial_epoch=init_epoch,
                verbose=1,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            temp_num = iter+start_model
            model.save_weights(log_dir + 'finalmodel_stage_'+str(temp_num)+'.h5')

    # Further training if needed.

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            #num = (185, len(model_body.layers)-3)[freeze_body-1]
            num = (180, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines,prefix, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], prefix ,input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines,prefix, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, prefix, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
