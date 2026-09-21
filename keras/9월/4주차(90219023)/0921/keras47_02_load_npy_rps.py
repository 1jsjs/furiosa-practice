
# acc 1.0 만들기
"""
* save
loss : 0.000346549553796649
acc : 1.0
accuracy score : 1.0
걸린 시간 : 1553.79 s

*load
loss : 0.000346549553796649
acc : 1.0
accuracy score : 1.0
"""
import time
import datetime
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 데이터 경로 설정 (horse-human 최상위 폴더 지정)
# ImageDataGenerator는 지정된 경로 아래의 하위 폴더(horses, humans)를 각각 클래스로 인식
path_data = "./_data/image/rps/"
path_save = "./_save/keras47_rps/"

np_path = "./_data/rps_npy/"
x_train = np.load (np_path + 'keras47_01_x_train.npy')
y_train = np.load (np_path + 'keras47_01_y_train.npy')
x_test = np.load (np_path + 'keras47_01_x_test.npy')
y_test = np.load (np_path + 'keras47_01_y_test.npy')

# 2. 모델 구성 + # 3. 컴파일, 훈련
model = load_model(path_save +"k47_0921_11150534-0.000370.keras") #모델체크포인트 파일에서 만들어놓은 모델 불러오기 모델구조 ~~ 모든 게 저장되어 있음

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)

y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print ('accuracy score :', round(acc_score, 2))