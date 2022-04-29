import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    ret, cam = cap.read()

    if(ret) :
        cv.imshow('camera', cam)
        
        
        if cv.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break

cap.release()
cv.destroyAllWindows()
