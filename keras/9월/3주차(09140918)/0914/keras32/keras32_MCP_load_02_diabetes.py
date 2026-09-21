# keras28_Scaler02_diabetes.py copy


from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time

path = "./_save/keras31_diabetes/"


#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print (x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.80, random_state=333)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform (x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform (x_test)

#2.모델구성 (input_dim = 10, 행무시 열우선)
model = load_model(path +"k31_0914_13440012-2209.800049.keras") #모델체크포인트 파일에서 만들어놓은 모델 불러오기 모델구조 ~~ 모든 게 저장되어 있음



#3.컴파일, 훈련
# model.fit (x_train, y_train, epochs = 1000,
#            batch_size = 3, 
#            validation_split = 0.2,
#            callbacks = [es, mcp])


#4.평가, 예측
loss = model.evaluate (x_test, y_test)
print ('loss : ', loss) #loss :  2690.05908203125

y_predict = model.predict (x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score (y_test, y_predict)
print ("r2 :", r2) #r2 : 0.5057520187712794

loss = model.evaluate (x_test, y_test)
print ('loss :', loss)

"""
save 파일 = load 파일 [성공]
loss :  3328.781005859375
r2 : 0.3721537991451218
loss : 3328.781005859375
"""