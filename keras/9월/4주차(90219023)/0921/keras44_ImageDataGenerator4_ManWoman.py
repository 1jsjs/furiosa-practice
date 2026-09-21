#https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data
#가중치와 모델 save
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

# 수치화 + 데이터 증폭 => ImageDataGemerator

path_data = "./_data/image/men_women"
path_save = "./_save/keras44_MenWomen/"

xyall_dategen = ImageDataGenerator (
    rescale = 1./255,
    # horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True, # 수직 뒤집기 (상하반전)
    # width_shift_range = 0.1, # 평행 이동
    # height_shift_range = 0.1,
    # rotation_range = 5, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,
    # shear_range = 0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (한마디로 피부...?)
    # fill_mode='nearest'
)

xy_all = xyall_dategen.flow_from_directory (
    path_data, # 경로
    target_size = (300, 300),
    batch_size= 30000, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'categorical',
    color_mode='rgb', # 컬러
    shuffle = True
)
x = xy_all[0][0]
y = xy_all[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.9, 
    random_state=333, 
    stratify=y # 클래스 비율 유지
)



# 2. 모델 구성
model = Sequential()

model.add(Conv2D(256, (3,3), activation="relu", input_shape=(300, 300, 3), padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3,3), activation='relu', padding='same')) 
model.add(MaxPool2D())
model.add(Dropout(0.2))

# 분류기 (Classifier) 부분
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile (loss='binary_crossentropy', optimizer='adam', #이진분류!!!!
               metrics=['acc'])
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k44_", date, filename])
# 파일명 예
# "./_save/keras30/" + "k30_", 0914_1147, 에포수-val_loss의 소수 4번째자리까지.keras
mcp = ModelCheckpoint ( 
    monitor='val_loss',
    mode='auto',
    save_best_only = True,
    filepath =filepath,
    verbose=1,
)

es = EarlyStopping (
    monitor= 'val_loss', #기준을 선언
    mode= 'min', #어떤 값을 찾을까? 긴가민가 하면 auto 하면 됨
    patience= 100, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
    verbose = 1, 
)

start_time = time.time()
model.fit (x_train, y_train, epochs=5000,
           batch_size=128, verbose=1, validation_split=0.1, callbacks=[es, mcp])
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