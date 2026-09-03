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
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit (x_train, y_train, epochs = 2000, batch_size =10)

print ("=================================================")
#4.평가, 예측
loss = model.evaluate (x_test, y_test, ) #default batch_size = 32
# evaluate 를 풀어서 쓴다면?
#  y(예측값) = w * x + b
#             w * x_test + b
#      loss = Σ(y_i(test) - ŷ_i)^2 / n
# loss 할 때 x_test, y_test 둘다 쓴다.

#y^- = w(마지막 epoch 에서 나온 가중치 값으로) * x_test + b
print ('loss(mse) : ', loss) #loss :  28.8504581451416

results = model.predict (x_test) #y 예측값이라서 y_predict라고 써도 됨 (이해를 돕기 위해)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, results) # 인풋과 아웃풋을 다 알아야 한다.

print ("r2 :", r2)


#predict 에서도 w(마지막 epoch 에서 나온 가중치 값으로 구한다.

# r2 : 0.7226553409714727