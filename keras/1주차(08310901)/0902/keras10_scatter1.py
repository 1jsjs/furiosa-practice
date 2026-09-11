import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test 데이터를 섞어서 7:3 으로 나눈다. 힌트 : 사이킷런
# Scikit Learn을 활용한 train-test 데이터셋 나누기 (cross validation, K-fold)
# https://blog.naver.com/bosongmoon/221794935181
# https://brain-nim.tistory.com/39


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, #train_size 나 test_size 두 개 중 아무거나 써도 된다.
                                                                    #둘 다 써도 됨 둘의 사이즈를 1로 안 맞춰도 됨
                                                                    #size를 1을 넘으면 안 된다. 
                                                                    #test_size: 테스트 셋 구성의 비율을 나타냅니다. 
                                                                    # train_size의 옵션과 반대 관계에 있는 옵션 값이며, 
                                                                    # 주로 test_size를 지정해 줍니다. 
                                                                    # 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. 
                                                                    # default 값은 0.25 입니다.
                                                    shuffle=True,
                                                    random_state=311, #매번 랜덤으로 뽑지 못 하게 시드 설정
                                                    )

print (x_train, x_test)
print (y_train, y_test)

#2.모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3.컴파일 훈련
model.compile (loss = 'mse', optimizer = 'adam')
model.fit (x_train ,y_train, epochs = 100, batch_size = 5)
print ("=================================================")

#4.평가 예측
loss = model.evaluate (x_test, y_test) #evaluate 는 딱 한번 epoch 함
print ('loss :', loss) #loss : 1.567646336297912e-06

results = model.predict(x)
print ('11의 결과 :', results) #11의 결과 : [[11.001989]]

#시각화 (그래프 그리기)
import matplotlib.pyplot as plt
plt.scatter (x, y)#데이터 점 찍기
plt.plot(x, results,color='red')
plt.show()