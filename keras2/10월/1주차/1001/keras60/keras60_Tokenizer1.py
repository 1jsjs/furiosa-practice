# 임베딩 한다 = 수치화 한다 = #1 데이터에서 한다 (전처리과정임)
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import  Tokenizer
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.utils import to_categorical

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."

token = Tokenizer() #인스턴스(객체) = 클래스 (), 인스턴스 생성
token.fit_on_texts([text])

print (token.word_index) 
# 빈도 순
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}

print (token.word_counts)
#OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print (len(x)) #[[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]

x = np.array(x).ravel()
print (x.shape) #(14,)

#원핫인코딩해야함
##1.pandas
# x = pd.get_dummies(x)
# print(x)
# print (x.shape) #(14, 9)

##2.sklearn
# x = to_categorical(x)
# print(x)
# print (x.shape) #(14, 10)

##3.keras
ohe = OneHotEncoder(sparse_output=False) #sparse 형태로 나온다.
x = x.reshape(-1, 1)
x = ohe.fit_transform(x)
print (x)
print (x.shape)