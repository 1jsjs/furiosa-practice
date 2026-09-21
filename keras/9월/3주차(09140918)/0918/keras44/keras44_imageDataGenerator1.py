import time
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

# 수치화 + 데이터 증폭 => ImageDataGemerator

path_train = "./_data/image/brain/train"
path_test = "./_data/image/brain/test"

train_dategen = ImageDataGenerator (
    rescale = 1./255,
    horizontal_flip = True, # 수평 뒤집기
    vertical_flip = True, # 수직 뒤집기 (상하반전)
    width_shift_range = 0.1, # 평행 이동
    height_shift_range = 0.1,
    rotation_range = 5, # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 1.2,
    shear_range = 0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (한마디로 피부...?)
    fill_mode='nearest'
)

test_dategen = ImageDataGenerator (
    rescale = 1./255,
)

xy_train = train_dategen.flow_from_directory (
    path_train, # 경로
    target_size = (100, 100),
    batch_size= 10, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode= "grayscale", # 흑백
    shuffle = True
)
# Found 160 images belonging to 2 classes.


xy_test = test_dategen.flow_from_directory (
    path_test,
    target_size = (100, 100),
    batch_size= 10, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'binary', # 이진분류라는 얘기
    color_mode= "grayscale", # 흑백
    # shuffle = True # 테스트 데이터는 건들지 않는 것이 원칙이므로 shuffle을 할 필요가 없다.
)
# Found 120 images belonging to 2 classes.
"""
print (xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001C30E8C7F70>

think : python Iterator
https://www.reddit.com/r/learnpython/comments/8o5ys1/what_is_an_iterator_please_can_you_explain_it/?tl=ko
print (xy_train.next()) # 이터레이터의 첫번째를 보여줘
print (xy_train.next()) # 두번째 이터레이터를 출력해줘!!!

print (xy_train[0][0]) # 첫번째 배치의 x데이터가 출력됨
print (xy_train[0][1]) # 첫번째 배치의 y데이터가 출력됨
print (xy_train[0][0].shape) # (10, 100, 100, 1)
print (xy_train[0][1].shape) # (10,)

print (xy_test[8][0])
print (xy_train[16][0]) # 여기서부터 엘. 이유는 160장이기에 . 배치가 10이여서!!

print (type(xy_train[0])) #<class 'tuple'>
print (type(xy_train[0][0])) #<class 'numpy.ndarray'>
"""

