import cv2 #import cv2 library
import numpy as np
#img = cv2.imread("/home/pi/Raspberrypi_workshop/Face_Mask_Detection_Machine_Learning/Human_face.jpeg") #load image

#to start the camera
capture = cv2.VideoCapture(0)


#load xml feature file so that we can detect face from image 
haar_data = cv2.CascadeClassifier("/home/pi/Raspberrypi_workshop/Face_Mask_Detection_Machine_Learning/haarcascade_frontalface_default.xml")

#collect all faces into this data 
data =[]
while(1):
    #load the image from camera
    flag,img = capture.read()
    #check camera is working or not
    if flag :
        #this function return array for faces from image 
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            #draw rectangle over face
            #cv2.rectangle(img,(x,y),(w,h),(r,g,b),boarder_thickness)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            #slice the face fro image
            face = img[y:y+w,x:x+w,:]
            #resize the face into one size
            face = cv2.resize(face,(50,50))
            #collect only 200 faces
            print(len(data))
            if(len(data) < 100):
                #store faces into dataset
                data.append(face)
        cv2.imshow("output",img) #show image
        # 27 is ASCII value of escapse key
        #It is use to run thread which waiting for ESC button becuase of this image will show on window
        #if you not use this function then it will not load image
        if((cv2.waitKey(2) == 27)):
            #save data into file
            np.save("with_mask.npy",data)
            break;
cv2.destroyAllWindows()
