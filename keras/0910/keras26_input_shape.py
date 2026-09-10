# keras23_softmax1_OneHot_iris.py copy

import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_iris()
###################################################원핫 1. to_categorical############################################
# from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)
# print (y)
# print (y.shape)
###################################################원핫 2. in pandas#################################################
# https://mizykk.tistory.com/13
# print (y)
# y = pd.get_dummies(datasets['target'])
# print (y)
# print (y.shape)
###################################################원핫 3. in sklearn#################################################
# https://mizykk.tistory.com/12
# https://tnqkrdmssjan.tistory.com/86
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) #sparse 형태로 나온다.
y = ohe.reshape(-1, 1) #데이터 전체를 뜻하는 거 -1 ? //데이터의 가장 마지막을 알아야 할 때 -1을 표시하면 가장 끝 값이 나온다. 
y = ohe.fit_transform(y)
print (y)
######################################################################################################################

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.8,
                                                     random_state=42,
                                                     shuffle=True, stratify=y) 
#shuffle은 true로 잡아야 한다. 시계열 문제 빼고 // stratify는 데이터 분포 비율을 맞추어 주는 파라미터 

print (x_train.shape, x_test.shape) #(120, 4) (30, 4)
print (y_train.shape, y_test.shape) #(120, 3) (30, 3)

#2.모델구성
model = Sequential()
# model.add (Dense(10, input_dim=4, activation='relu'))
model.add (Dense(10, input_shape=(4,), activation='relu')) #input_shape
model.add (Dense(10,activation='relu'))
model.add (Dense(10,activation='relu'))
model.add (Dense(10,activation='relu'))
model.add (Dense(3,activation='softmax'))
# softmax(다 더해서 그걸로 n빵)를 통해 가장 큰 값을 1로 주고 나머지를 0으로 준다.
"""
원 데이터       input_shape
(n,4)          데이턱 이렇게마녀 (4,)
(n, 100, 3) -> 행무시하면  -> (100, 3) 
 |
 행
 (n, 100, 100, 3) -> 행무시 -> (100, 100, 3)
 이 이상 shape는 보기는 어렵다.
"""

#3.컴파일, 훈련
model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping (
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
)
start_time = time.time()
model.fit (x_train, y_train, epochs = 1000, batch_size = 8,
           verbose=1, 
           validation_split=0.2,
           callbacks=[es],
           )
end_time = time.time()


#4.평가, 예측
results = model.evaluate (x_test, y_test)
print ('loss :', results[0]) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc
print ('acc :', round(results[1], 2)) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc // acc는 값이 길게 나오기 때문에 반올림 ㄱㄱ

y_pred = model.predict(x_test) # 와이 햇에 w * x_test + b 한 값을 넣은 것 뿐이다.
# print (y_pred)
y_pred = np.argmax(y_pred, axis=1) #[2 0 1 1 2 1 2 2 0 2 0 0 0 1 2 2 2 1 1 0 2 0 1 0 0 1 1 2 0 2]
# print (y_pred)
y_test = np.argmax(y_test, axis=1) #[2 0 1 1 2 1 2 2 0 2 0 0 0 1 2 2 2 1 1 0 2 0 1 0 0 1 1 2 0 2]
# print (y_test) #[2 0 1 1 2 1 2 2 0 1 0 0 0 1 2 2 2 1 1 0 2 0 1 0 0 1 1 2 0 2]

accuracy_score = accuracy_score(y_test, y_pred)
print ('acc_score : ', accuracy_score)
print ('걸린 시간 : ', round(end_time - start_time, 2), 's')