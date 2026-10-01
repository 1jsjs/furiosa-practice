#61-3 copy
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
from tensorflow.keras.layers import Dense, Embedding
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

#2. 모델
model = Sequential()
############################# 임베딩 1 #############################
model.add(Embedding(input_dim=30,   # input_dim은 단어사전의 개수라고 생각하면 된다.
                                    # 그럼 토큰 번호가 1~30이고 0은 패딩용이라서 input_dim=31이 맞아요. 
                                    # 인덱스가 0부터 시작하므로, 30번 토큰을 넣으려면 크기가 30 + 1 = 31이어야 합니다. 
                                    # input_dim은 토큰 개수가 아니라 사용할 수 있는 인덱스 범위의 크기예요.

                    output_dim=100, # 차원의 수
                    input_length=5))
model.add(SimpleRNN(10))
model.add(Dense(1))
"""
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 5, 100)            3000      
                                                                 
 simple_rnn (SimpleRNN)      (None, 10)                1110

Param # = input_dim X output_dim (메모리 공간이라고 생각하는 게 편하다)
"""

############################# 임베딩 2  #############################
"""
model.add(Embedding(input_dim=30, output_dim=100,))
# 임베딩 레이어에서는 input_length 를 쓰지 않아도 알아서 맞춰준다.
model.add(SimpleRNN(10))
model.add(Dense(1))
"""

############################# 임베딩 3  #############################
"""
model.add(Embedding(30, 100,)) inputdim, outputdim은 파라미터 명 명시 안 해도 되지만 length는 안 된다.
# 임베딩 레이어에서는 input_length 를 쓰지 않아도 알아서 맞춰준다.
model.add(SimpleRNN(10))
model.add(Dense(1))
"""

model.summary()
exit()
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
model.fit (padded_x, labels,
                  epochs = 1000,
                  batch_size = 1,
                  validation_split = 0.1,
                  callbacks = [es], #EarlyStopping을 리스트로 받아드림. 
                  )
end_time = time.time()