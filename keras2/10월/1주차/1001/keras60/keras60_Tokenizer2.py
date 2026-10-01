# 60-1 copy
# 임베딩 한다 = 수치화 한다 = #1 데이터에서 한다 (전처리과정임)
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import  Tokenizer
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.utils import to_categorical

text1 = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길똥이는 마구 마구 더 잘생겼다."

token = Tokenizer() #인스턴스(객체) = 클래스 (), 인스턴스 생성
token.fit_on_texts([text1])
x1 = token.texts_to_sequences([text1])
# print (x1) #[[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]

token.fit_on_texts([text2])
x2 = token.texts_to_sequences([text2])
# print (x2) #[[11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

x1 = np.array(x1).ravel()
print (x1.shape)
x2 = np.array(x2).ravel()
print (x2.shape)

x = np.concatenate((x1, x2))

x = pd.get_dummies(x)
print(x)
print (x.shape) #(24, 15)