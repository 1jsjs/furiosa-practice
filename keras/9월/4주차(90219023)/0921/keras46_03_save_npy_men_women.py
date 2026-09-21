# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data
# 남녀 얼굴 이미지를 읽어서 train/test용 numpy 파일로 저장
"""
-keras44_ImageDataGenerator4_ManWoman.py- > 이거 할 필요없지 않나요? 해서 삭제

keras46_03_save_npy_men_women.py (데이터 넘파이 저장)

keras47_03_laod_npy_men_women.py (넘파이 데이터 불러와서 훈련 후, 가중치 저장)

keras49_02_meManWoman.py (넘파이와 가중치 불러와서 내 사진으로 predict)
-----------------------------------------------------------------------------------------
keras44
이미지로 남녀 모델 학습 + .keras 가중치 저장

keras46
학습/테스트 이미지를 .npy로 저장

keras49
.npy로 만든 내 사진 + keras44에서 저장한 .keras 모델을 불러와 predict
-----------------------------------------------------------------------------------------
*keras44
loss : 0.6501535773277283
acc : 0.6449999809265137
accuracy score : 0.65
걸린 시간 : 1782.59 s

*keras46
Found 27167 images belonging to 2 classes.
class_indices: {'man': 0, 'woman': 1}
x_train: (1800, 300, 300, 3) y_train: (1800, 2)
x_test : (200, 300, 300, 3) y_test : (200, 2)
saved to: ./_data/men_women_npy/
"""

import time
import datetime
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# keras44와 같은 원본 데이터 폴더
path_data = "./_data/image/men_women/"
np_path = "./_data/men_women_npy/"


# 이미지 크기와 정규화 방식을 keras44와 맞춘다.
datagen = ImageDataGenerator(rescale=1./255)

xy_all = datagen.flow_from_directory(
    path_data,
    target_size=(300, 300),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=2000,
    shuffle=True,
)

# 현재 데이터가 많으므로 한 번에 2,000장만 메모리에 올린다.
# keras44도 xy_all[0]으로 같은 방식의 데이터 묶음을 사용했다.
x, y = xy_all[0]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.9,
    random_state=333,
    stratify=y,
)

np.save(np_path + 'keras46_03_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_03_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_03_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_03_y_test.npy', arr=y_test)

print("class_indices:", xy_all.class_indices)
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test :", x_test.shape, "y_test :", y_test.shape)
print("saved to:", np_path)
