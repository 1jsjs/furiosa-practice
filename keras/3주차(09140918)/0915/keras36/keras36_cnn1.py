from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, # 커널 개수
                 (3,3), # 각 커널의 크기 
                 input_shape=(10,10,1))) # 입력 이미지 크기
model.add(Conv2D(5, #커널 개수
                 (2,2))) # 각 커널의 크기 

model.summary()
# 이거 계산 하는 거 내일 할거임 가중치, 필터 어떻게 되어 있는지 // 머리 쓰기 싫어 > 편안하게 or 완벽하게 하려면 > 내일 다 까주겠다
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 10)          100       
                                                                 
#  conv2d_1 (Conv2D)           (None, 7, 7, 5)           205       
                                                                 
# =================================================================
# Total params: 305
# Trainable params: 305
# Non-trainable params: 0