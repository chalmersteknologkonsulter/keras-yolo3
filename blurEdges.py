from PIL import Image, ImageDraw, ImageFilter
from random import randint
import csv
import numpy as np
import os
import glob
import copy


def blurImage(road1, sign, x, y):
    road = copy.copy(road1)
    road.paste(sign, (x, y))
    #road.show()
    roads = list()
    roads.append(road)
    #roads[-1].show()
    num_of_it = 4
    for bl in range(num_of_it):
        frac = (bl + 1) / 40
        marginx = round(sign.size[0] * frac)
        marginy = round(sign.size[1] * frac)
        left_side = roads[-1].crop((x - marginx, y - marginy, x + marginx, y + sign.size[1] + marginy))  # Left edge of transition
        right_side = roads[-1].crop((x + sign.size[0] - marginx, y - marginy, x + sign.size[0] + marginx,y + sign.size[1] + marginy))  # Left edge of transition
        top_side = roads[-1].crop((x - marginx, y - marginy, x + sign.size[0] + marginx, y + marginy))  # Left edge of transition
        bottom_side = roads[-1].crop((x - marginx, y + sign.size[1] - marginy, x + sign.size[0] + marginx,y + sign.size[1] + marginy))  # Left edge of transition

        left_side = left_side.filter(ImageFilter.GaussianBlur)
        #print("left")
        right_side = right_side.filter(ImageFilter.GaussianBlur)
        #print("right")
        top_side = top_side.filter(ImageFilter.GaussianBlur)
        #print("top")
        bottom_side = bottom_side.filter(ImageFilter.GaussianBlur)
        #print("bottom")

        roads[-1].paste(left_side, (x - marginx, y - marginy))
        roads[-1].paste(right_side, (x + sign.size[0] - marginx, y - marginy))
        roads[-1].paste(top_side, (x - marginx, y - marginy))
        roads[-1].paste(bottom_side, (x - marginx, y + sign.size[1] - marginy))
        #if bl % 5 == 0:
            #roads[-1].show()
        roads.append(roads[-1])
    return roads[-1]
