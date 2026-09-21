# acc 1.0 만들기
"""
loss : 0.000346549553796649
acc : 1.0
accuracy score : 1.0
걸린 시간 : 1553.79 s
"""
import time
import datetime
from matplotlib.animation import ImageMagickFileWriter
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ImageMagickFileWriter.LOAD_TRUNCATED_IMAGES = True 
"""
가위바위보 실습에서 잘린 이미지 때문에 에러가 발생하는데, 잘린 이미지가 많을 경우에는 일일이 다 제거하는 건 비효율적이라

ImageFile.LOAD_TRUNCATED_IMAGES = True

이 코드 추가하시면 이미지 삭제하지 않더라도 에러 발생하지 않네요!
"""

# 1. 데이터 경로 설정 (horse-human 최상위 폴더 지정)
# ImageDataGenerator는 지정된 경로 아래의 하위 폴더(horses, humans)를 각각 클래스로 인식
path_data = "./_data/image/rps/"
path_save = "./_save/keras47_rps/"

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
    batch_size= 2000, 
    # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
    class_mode= 'categorical', # 이진분류라는 얘기
    color_mode='rgb', # 컬러
    shuffle = True
)
# Found 160 images belonging to 2 classes.
# xy_test = test_dategen.flow_from_directory (
#     path_data,
#     target_size = (150, 150),
#     batch_size= 120, 
#     # IDG 하면  (80,100,100,1), (80,) 이렇게 뭉쳐서 나온다. 배치 사이즈를 10으로 주면 8 * (10, 100, 100, 1) 이런 식으로 나온다.
#     class_mode= 'binary', # 이진분류라는 얘기
#     color_mode='grayscale', # 흑백
#     # shuffle = True # 테스트 데이터는 건들지 않는 것이 원칙이므로 shuffle을 할 필요가 없다.
# )
# # Found 120 images belonging to 2 classes.

# print (xy_all)
x = xy_all[0][0]
y = xy_all[0][1]
# print (x.shape) # (1027, 300, 300, 1)
# print (y.shape) # (1027,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8, 
    random_state=333, 
    stratify=y # 클래스 비율 유지
)

# print (x_train.shape, y_train.shape) #(821, 300, 300, 1) (821,)
# print (x_test.shape, y_test.shape) #(206, 300, 300, 1) (206,)

np_path = "./_data/rps_npy/"
np.save(np_path + 'keras47_01_x_train.npy', arr=x_train) # x_train / arr=xy_train[0][0]도 가능
np.save(np_path + 'keras47_01_y_train.npy', arr=y_train) # y_train
np.save(np_path + 'keras47_01_x_test.npy', arr=x_test) # x_test
np.save(np_path + 'keras47_01_y_test.npy', arr=y_test) #y_test

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(300,300,3),padding='same', strides=2)) 
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu',padding='same', strides=2)) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', strides=2))
model.add(MaxPool2D())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', strides=2)) 
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=8, kernel_size=(2,2), activation='relu', padding='same', strides=2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(3, activation='softmax')) # 이진분류는 sigmoid !!!!!!!!

# 3. 컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam', #이진분류!!!!
               metrics=['acc'])

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:4f}.keras' #history에서 때오는 것임
filepath = "".join([path_save, "k47_", date, filename])

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
           batch_size=4, verbose=1, validation_split=0.1, callbacks=[es, mcp])
end_time = time.time()

# 4. 평가, 예측
print ('==============model.evaluate===================')
loss = model.evaluate (x_test, y_test, verbose=1)
print ('loss :', loss[0])
print ('acc :', loss[1])

y_pred = model.predict (x_test)
y_pred_argmax = np.argmax(y_pred, axis=1)
y_test_argmax = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_argmax, y_pred_argmax)
print ('accuracy score :', round(acc_score, 2))
print ('걸린 시간 :', round(end_time - start_time, 2), 's')