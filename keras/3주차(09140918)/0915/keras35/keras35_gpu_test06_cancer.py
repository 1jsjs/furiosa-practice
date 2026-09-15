import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer #유방암관련 데이터셋 불러오기
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

"""
AI 는 두 종류가 있다.

        --- 다중
        |
_ 분류   |
|       |
|       --- 이진 : stratify, sigmoid, loss = 'binary_crossentropy'
|
|_ 회귀

"""
#1.데이터
#print(datasets.DESCR) #describe 약자 / 실무에선 쓸 수 읎다.
# Attribute Information > columns 명
# attribute 30개 , instance는 데이터의 개수? 열의 개수? => (569, 30) => input_dim = 30

datasets = load_breast_cancer () #딕셔너리 형식 데이터셋이다.
x = datasets['data']
y = datasets['target']

# x = datasets.data
# y = datasets.target
# print (x.shape, y.shape) #(569, 30) (569,)
# print (type(x),type(y)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print (y) # 0과 1의 개수가 몇 개인지 찾아보기 -
# print (np.unique(y)) #[0 1] => y 데이터 중에 독특한 놈은 0하고 1 밖에 없다 > 이진분류이다.
# print (np.unique(y, return_counts=True)) 
# #[0 1] => y 데이터 중에 독특한 놈은 0하고 1 밖에 없다 > 이진분류이다.
# # (array([0, 1]), array([212, 357])) > 0이 212개, 1이 357개. 분류모델은 분류할 것들이 골고루 있어야 성능 up
# print (pd.DataFrame(y).value_counts())
# # 1    357
# # 0    212
# print (pd.Series(y).value_counts())
# # 1    357
# # 0    212
# exit()

x_train, x_test, y_train, y_test = train_test_split (x, y, 
                                                     train_size=0.7,
                                                     random_state=42,
                                                     stratify = y # y를 기준으로 0과 1의 분포만큼 나눠준다.
                                                     )

# print (np.unique(y_train, return_counts=True))
# print (np.unique(y_test, return_counts=True))
# (array([0, 1]), array([149, 249])) > train 데이터가 불군형이라면 문제가 있다.
# (array([0, 1]), array([ 63, 108])) > 불균형이라도 상관없다. test는 훈련에 사용하지 않으니.

# print (x_train.shape, x_test.shape) # (398, 30) (171, 30)
# print (y_train.shape, y_test.shape) # (398,) (171,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform (x_train)
x_test = scaler.transform (x_test)



#2.모델구성
model = Sequential()
model.add (Dense(16, input_dim = 30, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #이진분류 마지막은 무조건 sigmoid // 0에서 1 사이 값으로 반환한다.
# https://velog.io/@happyyeon/ML-Logistic-Regression
# ŷ = sigmoid ( wx + b) <- 간략하면 이렇게.

#3.컴파일, 훈련
model.compile (loss = 'binary_crossentropy', optimizer = 'adam',  #mse는 수치화된 것에 대한 것. / 이진 분류에서는 딱 하나만 쓴다. **binary_crossentropy**
            #    metrics=['accuracy'],
               metrics=['acc'],
               )

es = EarlyStopping (
    monitor= 'val_loss', #기준을 선언
    mode= 'min', #어떤 값을 찾을까? 긴가민가 하면 auto 하면 됨
    patience= 20, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
)
start_time = time.time()
model.fit (x_train, y_train,
                  epochs = 1000,
                  batch_size = 32,
                  validation_split = 0.2,
                  callbacks = [es], #EarlyStopping을 리스트로 받아드림. 
                  )
end_time = time.time()

#4.평가, 예측
loss = model.evaluate (x_test, y_test)
print ("==============================================")
print ('loss :', loss[0]) 
print ('acc : ', round(loss[1],4))
print ("걸린 시간:", round(end_time-start_time, 2))
y_pred = model.predict(x_test)
# print (y_pred[:10]) #y_test 값은 sigmoid를 통해 0에서 1사이 값은 나오지만 딱 0과 1로 떨어지지 않아 우리가 처리해줘야 함.
y_pred = np.round(y_pred)
# print (y_pred[:10]) 

from sklearn.metrics import accuracy_score #acc 검증

acc_score = accuracy_score(y_test, y_pred) # test는 0아님 1, pred 는 0과 1 사이라 오류 아니면 다 0 처리
# ValueError: Classification metrics can't handle a mix of binary and continuous targets
print ("acc_score :", acc_score)

# 26.09.10 기준 minmax scaler 적용
"""
loss : 0.10378947108983994
acc :  0.9649
걸린 시간: 6.63
acc_score : 0.9649122807017544
"""



# 26.09.11 기준 standard scaler 적용
"""
loss : 0.13029098510742188
acc :  0.9708
걸린 시간: 4.41
acc_score : 0.9707602339181286
"""

"""
# 26.09.11 기준 MaxAbsScaler 적용
loss : 0.0798686146736145
acc :  0.9825
걸린 시간: 7.03
acc_score : 0.9824561403508771
"""