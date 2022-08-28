import os
import cv2
import random
import pickle
import pandas as pd
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


# 파라미터
path = "myData" # 이미지 파일 담겨있는 폴더
labelFile = 'labels.csv' # 라벨명 담겨있는 폴더
batch_size_val = 50 
steps_per_epoch_val = 100
epochs_val =10
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2
###############################################################

# 이미지 임포트 해오기
count = 0
images = []
classNo = []
myList = os.listdir(path)
# print("Total classes Detected: ", len(myList))
noOfClasses = len(myList)
# print("Importing Classes")


for x in range(0, len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count +=1
print(" ")
# 이미지 넘파이 배열로 바꿔주기
images = np.array(images)
classNo = np.array(classNo)

# 데이터 나눠주기
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validationRatio)

# x_train = Array of images to train
# y_train = corresponding lass id

###################################################################

# 이미지의 id 갯수랑 라벨갯수 맞는지 확인 
print("Data Shapes")
print("Train", end="");print(x_train.shape, y_train.shape)
print("Validation", end="");print(x_val.shape, y_val.shape)
print("Test", end="");print(x_test.shape, y_test.shape)
assert(x_train.shape[0]==y_train.shape[0])
assert(x_val.shape[0]==y_val.shape[0])
assert(x_test.shape[0]==y_test.shape[0])
assert(x_train.shape[1:]==(imageDimesions))
assert(x_val.shape[1:]==(imageDimesions))
assert(x_test.shape[1:]==(imageDimesions))

######################################################################
# CSV File 읽기
data=pd.read_csv(labelFile)
print("data shape", data.shape, type(data))

# 모든 클래스의 이미지 파일 보여주기
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+"-"+row["Name"])
            num_of_samples.append(len(x_selected))
            
# 카테고리별 바차트
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# 전처리(그레이스케일, 표준화, 정규화)
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

x_train= np.array(list(map(preprocessing, x_train)))
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))
cv2.imshow("GrayScale Images", x_train[random.randint(0, len(x_train)-1)]) 

# 1차원 추가하여 쉐이프 변화 시키기
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# 이미지 데이터 증강
dataGen = ImageDataGenerator(width_shift_range= 0.1,
                                height_shift_range= 0.1,
                                zoom_range= 0.2,
                                shear_range=0.1,
                                rotation_range=10)

dataGen.fit(x_train)
batches = dataGen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

# 데이터 증강된 이미지 보여주기
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# 커스텀 CNN 모델 생성
def myModel():
    no_Of_Fileters = 60
    size_of_Filter = (5, 5) 
    
    size_of_Filters2 = (3, 3)
    size_of_pool=(2, 2)
    no_Of_Nodes = 500
    
    model = Sequential()
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) 
    
    model.add((Conv2D(no_Of_Fileters//2, size_of_Filters2, activation='relu')))
    model.add((Conv2D(no_Of_Fileters//2, size_of_Filters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(noOfClasses, activation='softmax'))
    
    # CNN model 컴파일
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 훈련
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, epochs= epochs_val, validation_data=(x_val, y_val), shuffle=1) 

# History 
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# 모델 피클파일로 저장
# pickle_out = open("./model_trained.txt", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()
# cv2.waitKey(0)

model.save('my_model.h5') 