"""
keras46_03_save_npy_men_women.py (데이터 넘파이 저장)

keras47_03_laod_npy_men_women.py (넘파이 데이터 불러와서 훈련 후, 가중치 저장)

keras49_02_meManWoman.py (넘파이와 가중치 불러와서 내 사진으로 predict)
"""
import time
import datetime
import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. numpy 데이터와 모델 저장 경로
path_save = "./_save/keras47_MenWomen/"
os.makedirs(path_save, exist_ok=True)

np_path = "./_data/men_women_npy/"
x_train = np.load (np_path + 'keras46_03_x_train.npy')
y_train = np.load (np_path + 'keras46_03_y_train.npy')
x_test = np.load (np_path + 'keras46_03_x_test.npy')
y_test = np.load (np_path + 'keras46_03_y_test.npy')

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu",
                 input_shape=(300, 300, 3), padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="sigmoid"))

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = path_save + "k47_" + date + "-{epoch:04d}-{val_loss:.6f}.keras"

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=100,
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=1000,
    # 300x300 이미지 + 첫 Conv2D 256개 필터에서는 128이 너무 큼
    # RTX 4060 Laptop GPU(약 5.4 GiB)에서 OOM을 피하기 위한 값
    batch_size=4,
    validation_split=0.1,
    callbacks=[es, mcp],
    verbose=1,
)
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)

y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', round(acc_score, 2))
print ('걸린 시간 :', round(end_time - start_time, 2), 's')
