# 48 copy
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "C:\\study\\_data\\image\\"

img = load_img (path + 'mypic.jpg', target_size = (150, 150))

# print (img) #<PIL.Image.Image image mode=RGB size=150x150 at 0x1F97F6C9360>
# plt.imshow(img)
# plt.show()
arr = img_to_array(img)
# print (arr.shape) #(150, 150, 3)
# print (type(arr)) #<class 'numpy.ndarray'>

#reshape 해야 함
#4차원이라면 (1,150,150,3)임

arr = arr.reshape (1, 150, 150, 3)
print (arr.shape)
# arr = np.expand_dims (arr, axis = 0) # 차원 증가
# print (arr.shape) #(1, 150, 150, 3)

np_path = "./_data/brain_npy/"

np.save(np_path + "keras48_me.npy", arr=arr)

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

it = datagen.flow (arr, 
            batch_size = 1)
print (it) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000002BABFB3D180>

print (it.next()) # 파이썬 3.10까지
print (next(it)) # 파이성 3.11 이후
print (next(it).shape) # (1,150,150,3)

fig, ax = plt.subplots (nrows = 1, ncols=5, figsize=(5,5))

for i in range (5):
    batch = next (it)
    print (batch.shape)
    batch = batch.reshape (150, 150, 3)


    ax[i].imshow(batch)
    ax[i].axis ('off')

plt.show() # 확인해보니 5장 중 3장이 못 쓰는 사진이라 ImageDataGenerator 이동안 이걸 하지 말라고 했던 거임
# 이 5장은 수치로는 다 다른 장이다. 라벨로는 다 똑같은 사진
