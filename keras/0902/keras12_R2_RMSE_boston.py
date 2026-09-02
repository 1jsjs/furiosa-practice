#11-3 copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing


#1.데이터
#이미 데이터셋에서 train, test가 분리되어 있음
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print (x_train.shape, x_test.shape) #(404, 13) (102, 13)
print (y_train.shape, y_test.shape) #(404,) (102,)

#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 100, batch_size =4)

print ("=================================================")

#4.평가, 예측
loss = model.evaluate (x_test, y_test, )
#y^- = w(마지막 epoch 에서 나온 가중치 값으로) * x_test + b
print ('loss : ', loss) #loss :  28.8504581451416

#predict 에서도 w(마지막 epoch 에서 나온 가중치 값으로 구한다.