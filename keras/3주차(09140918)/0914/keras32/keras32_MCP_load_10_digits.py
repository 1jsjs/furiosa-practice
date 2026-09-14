#acc 1 만들기
# acc : 0.99
# acc_score :  0.9916666666666667
"""
# 9월 10일  minmax scaler 적용함
loss : 0.03355453163385391
acc : 0.99
acc_score :  0.99166666
"""
"""
9월 11일  StandardScaler 적용함
loss : 0.13738790154457092
acc : 0.99 
acc_score :  0.9888888888888889
걸린 시간 :  17.06 s
"""
"""
9월 11일  MaxAbsScaler 적용함
loss : 0.04351558908820152
acc : 0.99
acc_score :  0.9888888888888889
걸린 시간 :  18.18 s
"""
import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

path = "./_save/keras47/"

#1. 데이터
datasets = load_digits()

print (datasets.DESCR)
"""
:Number of Instances: 1797
:Number of Attributes: 64
:Attribute Information: 8x8 image of integer pixels in the range 0..16.
:Missing Attribute Values: None
"""
print (datasets)

x = datasets.data
y = datasets['target']

print (x.shape)
print (y.shape)
print (np.unique(y, return_counts=True))
y = pd.get_dummies(y)
print (y.shape)

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.80,
                                                     random_state=333,
                                                     stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()

scaler.fit (x_train) #x값은 모두 민맥스 스케일러 할 준비를 해라
x_train = scaler.transform (x_train) #변환시키기
x_test = scaler.transform (x_test) #변환시키기

print (x)
print (np.min(x_train), np.max(x_train)) #0.0 1.0
print (np.min(x_test), np.max(x_test)) #0.0 2.6666666666666665


# exit()


#2. 모델 구성
model = load_model (path + "k47_0914_15040087-0.037893.keras")

#3. 컴파일, 훈련

#4. 평가, 예측
results = model.evaluate (x_test, y_test)
print ('loss :', results[0]) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc
print ('acc :', round(results[1], 2)) # 값이 두개 나온다. loss와 metrics의 acc // 리스트 형태로 나온다. 0번째가 loss, 1번째가 acc // acc는 값이 길게 나오기 때문에 반올림 ㄱㄱ

y_pred = model.predict (x_test)
# print (y_pred)
y_pred = np.argmax (y_pred, axis=1)
# print (y_pred)
y_test = np.argmax (y_test, axis=1)
# print (y_test)

accuracy_score = accuracy_score(y_test, y_pred)
print ('acc_score : ', accuracy_score)


"""
loss : 0.02042042650282383
acc : 0.99
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score :  0.9888888888888889
걸린 시간 :  17.05 s

loss : 0.02042042650282383
acc : 0.99
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score :  0.9888888888888889
"""