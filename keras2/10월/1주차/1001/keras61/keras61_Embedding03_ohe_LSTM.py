#61-2 copy
"""
[[0.01460084]
 [0.99995303]
 [0.9919851 ]]
loss : 0.9994732141494751
acc :  0.8
걸린 시간: 80.69
"""
import time
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
labels = np.array ([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0]) # y (15,)

token = Tokenizer()
token.fit_on_texts(docs)
print (token.word_index)
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
# '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
# '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print (x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]


########## 패딩 #########

padded_x = pad_sequences(x,
                        maxlen = 5, #최대 길이 결정
                        padding='pre', #padding='post'
                        # truncating ='pre' # 자르는 건 default가 앞
                        )
# print (padded_x)
print (padded_x.shape) #(15, 5)

ohe = OneHotEncoder(sparse_output=False) #sparse 형태로 나온다.
x = padded_x.reshape(-1, 1)
x = ohe.fit_transform(x)
print (x)
x = x.reshape (-1, 5, 31)
print (x)
print (x.shape) #(15, 5, 31)


x_train, x_test, y_train, y_test = train_test_split (x, labels, 
                                                     train_size=0.7,
                                                     random_state=42,
                                                     stratify = labels # y를 기준으로 0과 1의 분포만큼 나눠준다.
                                                     )

print (x_train.shape, y_train.shape) #(10, 5, 31) (10,)
print (x_test.shape, y_test.shape) #(5, 5, 31) (5,)



print (x_train.shape, x_test.shape) #(10, 5, 31) (5, 5, 31)

######## 2. 모델구성
model = Sequential()
model.add (SimpleRNN(units=10, input_shape = (5,31))) 
# model.add(Dense(7, activation='relu'))
# model.add(GRU(10, input_shape=(3,1)))
# model.add(LSTM(10, input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일, 훈련
model.compile (loss = 'binary_crossentropy', optimizer = 'adam',  #mse는 수치화된 것에 대한 것. / 이진 분류에서는 딱 하나만 쓴다. **binary_crossentropy**
            #    metrics=['accuracy'],
               metrics=['acc'],
               )

es = EarlyStopping (
    monitor= 'val_loss', #기준을 선언
    mode= 'min', #어떤 값을 찾을까? 긴가민가 하면 auto 하면 됨
    patience= 200, #몇 번을 찾을 건인지
    restore_best_weights=True, #어떤 가중치 값을 반환할건지 default는 False *현재 값 / True는 최솟값
)
start_time = time.time()
model.fit (x_train, y_train,
                  epochs = 1000,
                  batch_size = 1,
                  validation_split = 0.1,
                  callbacks = [es], #EarlyStopping을 리스트로 받아드림. 
                  )
end_time = time.time()

#4.평가, 예측
loss = model.evaluate (x_test, y_test)

x_predict = ['개똥이 잘생겼다', '참 최고에요', '연기가 어색해요']

x_predict_seq = token.texts_to_sequences(x_predict)
x_predict_pad = pad_sequences(
    x_predict_seq,
    maxlen=5,
    padding='pre'
)

# 학습 때 fit한 같은 ohe를 사용
x_predict_ohe = ohe.transform(x_predict_pad.reshape(-1, 1))
x_predict_ohe = x_predict_ohe.reshape(
    -1, 5, x.shape[2]
)

print(x_predict_ohe.shape)  # (3, 5, 31)

y_predict = model.predict(x_predict_ohe)

print ("==============================================")
print(y_predict)
print ('loss :', loss[0]) 
print ('acc : ', round(loss[1],4))
print ("걸린 시간:", round(end_time-start_time, 2))

