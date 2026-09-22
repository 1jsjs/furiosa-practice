# 50-1 copy
"""
Keras의 ImageDataGenerator에서 flow() 메소드는 메모리에 저장된 단일 이미지 또는 이미지 배열(NumPy array)을 입력받아 
지정된 증폭(Data Augmentation) 조건을 적용한 뒤, 배치(batch) 단위로 무한히 생성해내는 데이터 이터레이터(Iterator)를 반환하는 역할을 합니다.
"""
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

############# 여기부터 증폭 #################
datagen = ImageDataGenerator (
    rescale = 1./255,
    # horizontal_flip = True, # 수평 뒤집기 (좌우반전)
    vertical_flip = True, # 수직 뒤집기 (상하반전)
    width_shift_range = 0.1, # 평행 이동
    # height_shift_range = 0.1,
    rotation_range = 7, # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 0.5, # 0이 default
    # shear_range = 0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest'
)

augment_size = 100
# print (x_train.shape) #(60000, 28, 28)
# print (x_train[0].shape) #(28, 28)

# aaa = np.tile(x_train[0], augment_size) #무슨 명령어? -> 몰라 tile 같은 명령어임 타일처럼 붙이는...? 복사...?
# # 증폭하는 데이터긴 한데 데이터가 똑같아서 이 데이터에 대한 과적합이 생김
# print (aaa.shape) #(28, 2800) > 이게 문제???????
# aaa = aaa.reshape(-1, 28,28,1)
# print (aaa.shape) #(100, 28, 28, 1)
aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1) #무슨 명령어? -> 몰라 tile 같은 명령어임 타일처럼 붙이는...? 복사...?
# print (aaa.shape) #(100, 28, 28, 1)

xy_data = datagen.flow (np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
              np.zeros(augment_size),
              batch_size=augment_size,
              shuffle=False,
              ).next()
# print (xy_data)
# print (type(xy_data)) #AttributeError: 'tuple' object has no attribute 'shape'
print (len(xy_data)) # 2 , x와 y 두개니까
print (xy_data[0].shape) #(100, 28, 28, 1)
print (xy_data[1].shape) #(100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1) #subplot은 0부터 시작하니에
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()