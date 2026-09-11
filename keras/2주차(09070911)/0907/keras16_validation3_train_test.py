#16-vaild2 copy
# train_test_split 으로 자르기
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# 8 4 4 로 나누기
#일단 트레인 - 테스트를 반반으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, random_state=42)

#테스트를 test-val를 반반으로 분리
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size = 0.5, random_state=42)

# print (x_train.shape, x_val.shape, x_test.shape) #(8,) (4,) (4,)
# print (y_train.shape, y_val.shape, y_test.shape) #(8,) (4,) (4,)
# print (x_train)
# print (x_val)
# print (x_test)
# exit()

#2.모델구성
model = Sequential()
model.add (Dense(10,input_dim = 1))
model.add (Dense(10))
model.add (Dense(10))
model.add (Dense(10))
model.add (Dense(1))

#3.컴파일, 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 50, batch_size = 4, 
           verbose = 1,
           validation_data = (x_val, y_val),
           )
#loss 값이 내려가고 있어 val loss 도 잘 내려간다  > 에포를 늘린다
#loss 값이 내려가고 있어 val loss 가 중간에서 왔다갔다  > 그 중간이 최적?
#loss 값이 내려가고 있어 val loss 가 더 내려간다  > loss 말고 val loss 를 믿어라

#4.평가, 예측
loss = model.evaluate (x_test, y_test) 
print ('loss :', loss)