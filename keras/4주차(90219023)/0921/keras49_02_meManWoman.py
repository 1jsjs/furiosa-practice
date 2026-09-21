"""
keras44_ImageDataGenerator4_ManWoman.py
    남녀 얼굴 모델 학습 + 가중치 저장

keras46_03_save_npy_men_women.py
    남녀 이미지 데이터를 numpy로 저장

keras49_02_meManWoman.py
    numpy와 keras44에서 저장한 가중치를 불러와 내 사진 predict

    ============== 내 사진 예측 결과 ==============
raw prediction : [[0.6416128  0.35843575]]
like man?   : 0.6416128
like woman? : 0.35843575
결과: man 쪽
"""
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


# keras46에서 저장한 테스트 데이터
np_path = "./_data/men_women_npy/"
x_test = np.load(np_path + "keras46_03_x_test.npy")
y_test = np.load(np_path + "keras46_03_y_test.npy")

# keras44에서 저장한 모델 중 val_loss가 가장 낮았던 모델
path_save = "./_save/keras44_MenWomen/"
model = load_model(path_save + "k44_0921_14500019-0.650756.keras")


# 테스트 데이터로 모델 성능 확인
print("============== model.evaluate ================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss :", loss[0])
print("acc  :", loss[1])

y_pred = model.predict(x_test, verbose=0)
y_pred_round = np.round(y_pred)
print("accuracy score :", round(accuracy_score(y_test, y_pred_round), 2))


# 내 사진을 남녀 모델 입력 크기인 300x300 RGB numpy로 변환
img = load_img("./_data/image/mypic.jpg", target_size=(300, 300))
x_me = img_to_array(img)
x_me = x_me.reshape(1, 300, 300, 3)
x_me = x_me / 255.0

# 내 사진 predict
print("============== 내 사진 예측 결과 ==============")
me_pred = model.predict(x_me, verbose=0)
print("raw prediction :", me_pred)
print("like man?   :", me_pred[0][0])
print("like woman? :", me_pred[0][1])

if np.argmax(me_pred[0]) == 0:
    print("결과: man 쪽")
else:
    print("결과: woman 쪽")
