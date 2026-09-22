"""기존 men/women npy를 불러와 여성 데이터만 증폭하여 학습한다.
loss: 1.6884045600891113
acc : 0.6800000071525574
"""

import numpy as np
import time
import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

path_save = "./_save/keras51_MenWomen/"
# 1. npy 파일 불러오기
np_path = "./_data/men_women_npy/"
x_train = np.load (np_path + 'keras46_03_x_train.npy')
y_train = np.load (np_path + 'keras46_03_y_train.npy')
x_test = np.load (np_path + 'keras46_03_x_test.npy')
y_test = np.load (np_path + 'keras46_03_y_test.npy')

print("원본 x_train:", x_train.shape) #원본 x_train: (1800, 300, 300, 3)
print("원본 y_train:", y_train.shape) #원본 y_train: (1800, 2)
print("원본 x_test :", x_test.shape) #원본 x_test : (200, 300, 300, 3)
print("원본 y_test :", y_test.shape) #원본 y_test : (200, 2)


# 2. np.where로 여성 데이터만 선택
# class_indices: {'man': 0, 'woman': 1}
# y_train의 shape은 (샘플 수, 2)이고 여성 라벨은 [0, 1]이다.
# 따라서 열 전체가 아니라 여성에 해당하는 두 번째 열만 확인한다.
women_idx = np.where(y_train[:, 1] > 0.0)[0]

x_train_woman = x_train[women_idx].copy()
y_train_woman = y_train[women_idx].copy()

print("여성 데이터 개수:", len(women_idx))
print("x_train_woman:", x_train_woman.shape)
print("y_train_woman:", y_train_woman.shape)

# 3. 여성 데이터만 증폭
augment_size = 1000

# 여성 데이터 안에서만 랜덤 선택한다.
randidx = np.random.randint(
    x_train_woman.shape[0],
    size=augment_size,
)

x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()

"""
x_train_woman = x_train[np.where(y_train > 0.0)]
y_train_woman = y_train[np.where(y_train > 0.0)]

print (x_train_woman.shaoe, y_train_woman.shape)
print (np.unique(y_train_woman, return_counts = True))
"""

# npy의 x 데이터는 이미 0~1이므로 rescale을 넣지 않는다.
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

augmented_generator = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
)
x_augmented, y_augmented = next(augmented_generator)

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

print("증폭 후 x_train:", x_train.shape)
print("증폭 후 y_train:", y_train.shape)


# 4. 모델 구성
model = Sequential()
model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(300, 300, 3), padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
model.summary()


# 5. 학습
date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = path_save + "k51_" + date + "-{epoch:04d}-{val_loss:.6f}.keras"


mcp = ModelCheckpoint(filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
es = EarlyStopping(monitor="val_loss", mode="min", patience=30, restore_best_weights=True, verbose=1)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=4,
    validation_split=0.1,
    callbacks=[es, mcp],
    verbose=1,
)
end_time = time.time()


# 6. 평가
print("============== model.evaluate ==============")
loss, acc = model.evaluate(x_test, y_test, verbose=1, callbacks = [es, mcp])
print("loss:", loss)
print("acc :", acc)

y_pred = model.predict(x_test, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)
print("accuracy score:", round(accuracy_score(y_test_class, y_pred_class), 2))
print("걸린 시간:", round(end_time - start_time, 2), "s")

