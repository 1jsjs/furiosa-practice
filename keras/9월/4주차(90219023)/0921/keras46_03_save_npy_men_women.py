# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data
# 남녀 얼굴 이미지를 읽어서 train/test용 numpy 파일로 저장
"""
-keras44_ImageDataGenerator4_ManWoman.py- > 이거 할 필요없지 않나요? 해서 삭제

keras46_03_save_npy_men_women.py (데이터 넘파이 저장)

keras47_03_laod_npy_men_women.py (넘파이 데이터 불러와서 훈련 후, 가중치 저장)

keras49_02_meManWoman.py (넘파이와 가중치 불러와서 내 사진으로 predict)
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
import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 현재 파일은 C:/study/keras/9월/4주차(90219023)/0921에 있고,
# 실제 _data 폴더는 C:/study/_data에 있다.
current_dir = os.path.dirname(os.path.abspath(__file__))
study_dir = os.path.normpath(os.path.join(
    current_dir, "..", "..", "..", ".."
))
path_data = os.path.join(study_dir, "_data", "image", "men_women")
np_path = os.path.join(study_dir, "_data", "men_women_npy")
os.makedirs(np_path, exist_ok=True)


# 이미지 크기와 정규화 방식을 keras44와 맞춘다.
datagen = ImageDataGenerator(rescale=1./255)

xy_all = datagen.flow_from_directory(
    path_data,
    target_size=(300, 300),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
)

# 전체 이미지의 인덱스를 train/test로 분리한다.
# 이미지 전체를 RAM에 올리지 않고 인덱스만 분리한다.
total_count = xy_all.samples
all_indices = np.arange(total_count)
class_ids = xy_all.classes

train_idx, test_idx = train_test_split(
    all_indices,
    train_size=0.9,
    random_state=333,
    stratify=class_ids,
)

train_position = np.full(total_count, -1, dtype=np.int64)
test_position = np.full(total_count, -1, dtype=np.int64)
train_position[train_idx] = np.arange(len(train_idx))
test_position[test_idx] = np.arange(len(test_idx))

# open_memmap은 배열을 한꺼번에 RAM에 만들지 않고 .npy 파일에 배치로 기록한다.
x_train = np.lib.format.open_memmap(
    os.path.join(np_path, "keras46_03_x_train.npy"),
    mode="w+",
    dtype="float32",
    shape=(len(train_idx), 300, 300, 3),
)
x_test = np.lib.format.open_memmap(
    os.path.join(np_path, "keras46_03_x_test.npy"),
    mode="w+",
    dtype="float32",
    shape=(len(test_idx), 300, 300, 3),
)

# 라벨은 이미지 전체를 만들지 않고 class id에서 바로 원-핫으로 만든다.
y_all = np.eye(len(xy_all.class_indices), dtype="float32")[class_ids]
y_train = y_all[train_idx]
y_test = y_all[test_idx]

for batch_no in range(len(xy_all)):
    x_batch, _ = xy_all[batch_no]
    start = batch_no * xy_all.batch_size
    end = min(start + len(x_batch), total_count)
    batch_indices = np.arange(start, end)

    train_mask = np.isin(batch_indices, train_idx)
    test_mask = ~train_mask

    if np.any(train_mask):
        x_train[train_position[batch_indices[train_mask]]] = x_batch[train_mask]

    if np.any(test_mask):
        x_test[test_position[batch_indices[test_mask]]] = x_batch[test_mask]

    if (batch_no + 1) % 50 == 0 or batch_no == len(xy_all) - 1:
        print(f"저장 진행: {end}/{total_count}")

x_train.flush()
x_test.flush()

print("class_indices:", xy_all.class_indices)
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test :", x_test.shape, "y_test :", y_test.shape)
print("saved to:", np_path)
