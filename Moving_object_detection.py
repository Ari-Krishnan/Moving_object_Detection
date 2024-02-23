import cv2
import time
import imutils

cam = cv2.VideoCapture(0) # Initializing the camera

firstframe = None 
area = 500 

while True:
    _,img = cam.read() # getting frames in camera
    text = "Normal" 
    img = imutils.resize(img, width = 500) # resize the window 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting color image to gray scale image
    gaussianblur = cv2.GaussianBlur(gray, (21, 21), 0) # Appling smoothening
    if firstframe is None: 
        firstframe = gaussianblur # saving the first tecked the plane backgroud image
        continue
    diffimg = cv2.absdiff(firstframe,gaussianblur)
    # differnce between to the firstframe and gaussianblur
    threshold = cv2.threshold(diffimg, 25, 255, cv2.THRESH_BINARY)[1]
    # Appling threshold on the gaussianblur image
    threshold = cv2.dilate(threshold, None, iterations = 2)
    cont = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    for c in cont:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h,) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Drawing rectangle
        text = "Moving Object Detaction"
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 120), 3)
    cv2.imshow("Streaming",img)
    key = cv2.waitKey(10)
    print(key)
    if key == 27: # the while loop excite until press Esc key 
        break
cam.release()
cv2.destroyAllWindows()


    













    
    
    
