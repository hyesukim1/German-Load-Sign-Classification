import numpy as np
import cv2
from keras.models import load_model

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# import the trained model


# with open("model_trained.txt", "rb") as pickle_in: ## rb = read byte
#     model = pickle.load(pickle_in)

model= load_model('my_model.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo):
    if   classNo == 0: return 'speed limit 20 km/h'
    elif classNo == 1: return
    elif classNo == 2: return
    elif classNo == 3: return
    elif classNo == 4: return
    elif classNo == 5: return
    elif classNo == 6: return
    elif classNo == 7: return
    elif classNo == 8: return
    elif classNo == 9: return
    elif classNo == 10: return
    elif classNo == 11: return
    elif classNo == 12: return
    elif classNo == 13: return
    elif classNo == 14: return
    elif classNo == 15: return
    elif classNo == 16: return
    elif classNo == 17: return
    elif classNo == 18: return
    elif classNo == 19: return
    elif classNo == 20: return
    elif classNo == 21: return
    elif classNo == 22: return
    elif classNo == 23: return
    elif classNo == 24: return
    elif classNo == 25: return
    elif classNo == 26: return
    elif classNo == 27: return
    elif classNo == 28: return
    elif classNo == 29: return
    elif classNo == 30: return
    elif classNo == 31: return
    elif classNo == 32: return
    elif classNo == 33: return
    elif classNo == 34: return
    elif classNo == 35: return
    elif classNo == 36: return
    elif classNo == 37: return
    elif classNo == 38: return
    elif classNo == 39: return
    elif classNo == 40: return
    elif classNo == 41: return
    elif classNo == 42: return

while True:
    
    # read image
    success, imgOrignal = cap.read()
    
    # process image
    img = np.array(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    #predict image
    predictions = model.predict(img)
    classIndex = model.predict_step(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        # print(getClassname(classindex))
        cv2.putText(imgOrignal, str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break