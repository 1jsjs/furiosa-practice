"""
개고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이ㅣ npy데이터 사용
내 사진도 npy 불러와서 predict만 하면 된다. 
----------------------------------------------------
============== 내 사진 예측 결과 ===================
[[0.499998 0.500002]] ---> ??????????????????
"""
import time
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# 1. 데이터 경로
np_path = "./_data/kaggle_cat_dog_npy/"  # 개·고양이 데이터 경로
me_path = "./_data/brain_npy/"           # 내 사진 데이터 경로

x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy')
x_me = np.load(me_path + 'keras48_me.npy')

x_me = x_me / 255.0

path_save = "./_save/keras44_CatDog/"
model = load_model(path_save + "k45_0921_13210001-0.693147.keras")

# 4. 테스트 데이터 평가 및 예측
print('============== model.evaluate ===================')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = model.predict(x_test)
y_pred_rounded = np.round(y_pred)
acc_score = accuracy_score(y_test, y_pred_rounded)
print('accuracy score :', round(acc_score, 2))

# 5. 내 사진 데이터 예측 (Predict)
print('============== 내 사진 예측 결과 ===================')
me_pred = model.predict(x_me)
x_me_rounded = np.round (me_pred)

print ('like cat?: ', me_pred[0][0])
print ('like dog?: ', me_pred[0][1])

if np.argmax(me_pred[0]) == 0:
    print("결과: cat 쪽")
else:
    print("결과: dog 쪽")