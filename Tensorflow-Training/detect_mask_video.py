# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
import spidev # To communicate with SPI devices

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

#***************************** Defining all use pin ***************************************#  

# Define GPIO to LCD mapping
LCD_RS = 7
LCD_E  = 11
LCD_D4 = 12
LCD_D5 = 13
LCD_D6 = 15
LCD_D7 = 16
green_led_pin = 29
dc_motor1 = 31
dc_motor2 = 32
Red_led_pin = 33
buzzer_pin = 36
GPIO.setup(LCD_E, GPIO.OUT)  # E
GPIO.setup(LCD_RS, GPIO.OUT) # RS
GPIO.setup(LCD_D4, GPIO.OUT) # DB4
GPIO.setup(LCD_D5, GPIO.OUT) # DB5
GPIO.setup(LCD_D6, GPIO.OUT) # DB6
GPIO.setup(LCD_D7, GPIO.OUT) # DB7
GPIO.setup(Red_led_pin, GPIO.OUT) 
GPIO.setup(dc_motor1, GPIO.OUT) 
GPIO.setup(dc_motor2, GPIO.OUT) 
GPIO.setup(green_led_pin , GPIO.OUT) 
GPIO.setup(buzzer_pin, GPIO.OUT)

#***************************** LCD Code start ***************************************#   

'''
define pin for lcd
'''
# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005
delay = 1

# Define some device constants
LCD_WIDTH = 16    # Maximum characters per line
LCD_CHR = True
LCD_CMD = False
LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x90# LCD RAM address for the 3nd line
LCD_LINE_4 = 0xD0# LCD RAM address for the 3nd line

    
'''
Function Name :lcd_init()
Function Description : this function is used to initialized lcd by sending the different commands
'''
def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)
'''
Function Name :lcd_byte(bits ,mode)
Fuction Name :the main purpose of this function to convert the byte data into bit and send to lcd port
'''
def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = data
  # mode = True  for character
  #        False for command
 
  GPIO.output(LCD_RS, mode) # RS
 
  # High bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x10==0x10:
    GPIO.output(LCD_D4, True)
  if bits&0x20==0x20:
    GPIO.output(LCD_D5, True)
  if bits&0x40==0x40:
    GPIO.output(LCD_D6, True)
  if bits&0x80==0x80:
    GPIO.output(LCD_D7, True)
 
  # Toggle 'Enable' pin
  lcd_toggle_enable()
 
  # Low bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x01==0x01:
    GPIO.output(LCD_D4, True)
  if bits&0x02==0x02:
    GPIO.output(LCD_D5, True)
  if bits&0x04==0x04:
    GPIO.output(LCD_D6, True)
  if bits&0x08==0x08:
    GPIO.output(LCD_D7, True)
 
  # Toggle 'Enable' pin
  lcd_toggle_enable()
'''
Function Name : lcd_toggle_enable()
Function Description:basically this is used to toggle Enable pin
'''
def lcd_toggle_enable():
  # Toggle enable
  time.sleep(E_DELAY)
  GPIO.output(LCD_E, True)
  time.sleep(E_PULSE)
  GPIO.output(LCD_E, False)
  time.sleep(E_DELAY)
'''
Function Name :lcd_string(message,line)
Function  Description :print the data on lcd 
'''
def lcd_string(message,line):
  # Send string to display
 
  message = message.ljust(LCD_WIDTH," ")
 
  lcd_byte(line, LCD_CMD)
 
  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)
    
#***************************** LCD  code end ***************************************#   
    
#***************************** Read Temperature code start ***************************************#    
# Start SPI connection
spi = spidev.SpiDev() # Created an o bject
spi.open(0,0)
spi.max_speed_hz = 1350000
# Read MCP3008 data
def analogInput(channel):  
  adc = spi.xfer2([1,(8+channel)<<4,0])
  data = ((adc[1]&3) << 8) + adc[2]
  return data
# Below function will convert data to voltage
def Volts(data):
  volts = (data * 5) / float(1023)
  volts = round(volts, 2) # Round off to 2 decimal places
  return volts
 
# Below function will convert data to temperature.
def Temp(data):
  temp = ((data * 500)/float(1023))
  temp = round(temp)
  return temp

#***************************** Read Temperature code end ***************************************#


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(0).start()

lcd_init()
lcd_string("welcome ",LCD_LINE_1)
time.sleep(2)
lcd_byte(0x01,LCD_CMD) # 000001 Clear display
lcd_string("Face Mask ",LCD_LINE_1)
lcd_string("Detection ",LCD_LINE_2)
time.sleep(2)
GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
GPIO.output(Red_led_pin,GPIO.LOW)  #LED OFF
GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        if(label == "Mask"):
            print("mask detected")
            lcd_byte(0x01,LCD_CMD) # 000001 Clear display
            lcd_string("Mask Detected ",LCD_LINE_1)
            time.sleep(1)
            temp_output = analogInput(0) # Reading from CH0
            temp_volts = Volts(temp_output)
            temp       = Temp(temp_output)
            lcd_byte(0x01,LCD_CMD) # 000001 Clear display
            lcd_string("Temperature ",LCD_LINE_1)
            lcd_string(str(temp),LCD_LINE_2)
            time.sleep(1)
            if(temp < 40):
                GPIO.output(green_led_pin,GPIO.HIGH)  #LED OFF
                GPIO.output(Red_led_pin,GPIO.LOW)  #LED OFF
                GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF
                lcd_byte(0x01,LCD_CMD) # 000001 Clear display
                lcd_string("gate open ",LCD_LINE_1)
                time.sleep(0.2)
                GPIO.output(dc_motor1,GPIO.HIGH)  #Motor OFF
                GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
                time.sleep(2)
                GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                GPIO.output(dc_motor2,GPIO.HIGH)  #MOTOR OFF
                time.sleep(2)
                GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
            else:
                GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
                GPIO.output(Red_led_pin,GPIO.HIGH)  #LED OFF
                GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
                GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer OFF
                lcd_byte(0x01,LCD_CMD) # 000001 Clear display
                lcd_string("High temperature ",LCD_LINE_1)
                time.sleep(1)
        elif(label == "No Mask"):
            GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
            GPIO.output(Red_led_pin,GPIO.HIGH)  #LED OFF
            GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
            GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
            GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer OFF
            print("mask not detected")
            lcd_byte(0x01,LCD_CMD) # 000001 Clear display
            lcd_string("please wear",LCD_LINE_1)
            lcd_string("mask",LCD_LINE_2)
            time.sleep(1)
        else:
            print("nothing")
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
          
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
