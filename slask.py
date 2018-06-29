h,w = 416,416
num_anchors = 9
num_classes = 3
import keras.backend as K
from keras.layers import Input, Lambda

y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
print(y_true[2].shape)
print(y_true)
