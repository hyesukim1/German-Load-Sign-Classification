from re import S
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

# parameters
path = "myData" # folder with all the class folders
labelFile = 'labels.csv' # file with all names of classes
batch_size_val = 50 # how manu to process together
steps_per_epoch_val = 2000
epochs_val =10
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2
###############################################################

# Importing of the Images
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
images = np.array(images)
classNo = np.array(classNo)

# split data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validationRatio)

# x_train = Array of images to train
# y_train = corresponding lass id

###################################################################

# to check id number of images matches to number of lables for each data set
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
# Read CSV File
data=pd.read_csv(labelFile)
print("data shape", data.shape, type(data))

# Display some samples images of all the classes
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
            
# Display a bar chart showing no of samples for each category
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# preprocessing the images
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
cv2.imshow("GrayScale Images", x_train[random.randint(0, len(x_train)-1)]) # to check if the training is done properly

# add a depth of 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Augmentation of images: to makeit more generic
dataGen = ImageDataGenerator(width_shift_range= 0.1,
                                height_shift_range= 0.1,
                                zoom_range= 0.2,
                                shear_range=0.1,
                                rotation_range=10)

dataGen.fit(x_train)
batches = dataGen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

# to show agmented image samples
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# convolution neural network model
def myModel():
    no_Of_Fileters = 60
    size_of_Filter = (5, 5) # this is the kernel that move around the image to get the features.
    
    size_of_Filters2 = (3, 3)
    size_of_pool=(2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    model.add((Conv2D(no_Of_Fileters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # does not effect the depth/no of filters
    
    model.add((Conv2D(no_Of_Fileters//2, size_of_Filters2, activation='relu')))
    model.add((Conv2D(no_Of_Fileters//2, size_of_Filters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5)) # inputs nodes to drop with each update 1 all 0 none
    model.add(Dense(noOfClasses, activation='softmax')) # output layer
    
    # compile model
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# train
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, epochs= epochs_val, validation_data=(x_val, y_val), shuffle=1) 

# plot
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

# store the model as a pickle object
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
