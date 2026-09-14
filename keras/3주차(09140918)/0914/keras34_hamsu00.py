from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model #함수형 모델이라고 선언하는 것
from tensorflow.keras.layers import Dense, Dropout, Input


#1. 데이터

#2-1. 순차적모델구성
model = Sequential()
#             1-1          1-2
model.add(Dense(10, input_shape=(3,)))
#               2
model.add (Dropout(0.2)) #이렇게 쓴다면 윗 줄에 적용되는거임
#               3
model.add(Dense(10))
#               4
model.add (Dropout(0.2)) #이렇게 쓴다면 윗 줄에 적용되는거임
#               5
model.add(Dense(1))

model.summary()
############################################################################# 2-1 과 2-2는 같은 거고 그냥 다르게 쓴 것 뿐
#2-2. 함수형모델구성
input1 = Input(shape=(3,)) # 1-2 한거임
dense1 = Dense(10,name='ys1')(input1) # 1-1 한거임 (input1)을 한 이유는 인풋과 덴스를 연결하기 위해
drop1 = Dropout(0.2)(dense1) # 2 한거임 / 
dense2 = Dense(10, name='ys2')(drop1) # 3 한거임 / 
drop2 =  Dropout(0.2)(dense2) # 4 한거임
output1 = Dense(1)(drop2) # 5 한거임
model2 = Model(inputs=input1, outputs=output1) # 모델 정의하기
model2.summary()
