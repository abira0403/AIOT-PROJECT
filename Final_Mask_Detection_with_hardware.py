#!/usr/bin/python
import time
import RPi.GPIO as GPIO
import time
import os,sys
from urllib.parse import urlparse
import paho.mqtt.client as paho
import spidev # To communicate with SPI devices
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
import cv2 #import cv2 library
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



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

#*****************************FaceMask training code start ***************************************#   

#load xml feature file so that we can detect face from image 
haar_data = cv2.CascadeClassifier("/home/pi/Raspberrypi_workshop/Face_Mask_Detection_Machine_Learning/haarcascade_frontalface_default.xml")
#load the data set file
with_mask = np.load("/home/pi/Raspberrypi_workshop/Face_Mask_Detection_Machine_Learning/with_mask.npy")
without_mask =np.load("/home/pi/Raspberrypi_workshop/Face_Mask_Detection_Machine_Learning/without_mask.npy")

#check the size of data_set
#print(with_mask.shape)
#print(without_mask.shape)

#convert data set into 2 diamentional array
with_mask = with_mask.reshape(100,50*50*3)
without_mask = without_mask.reshape(100,50*50*3)

#check the size of data_set
#print(with_mask.shape)
#print(without_mask.shape)

#combined two dataset
x = np.r_[with_mask,without_mask]
#print(x .shape)

#give label to your data
#now we have to label the dataset
#with_mask  = 0
#without_mask =1
label =np.zeros(x.shape[0])
label[100:]=1.0

x_train,x_test,y_train,y_test =train_test_split(x,label,test_size = 0.35)

svm = SVC()
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
#print the accuracy of your data
#accracy below 1 should be good
#print(accuracy_score(y_test,y_pred))

#create variant to display mask and no mask string 
names = {0 : "Mask", 1 :"No Mask"}

#font for name
font =cv2.FONT_HERSHEY_COMPLEX

#to start the camera
capture = cv2.VideoCapture(0)

#*****************************FaceMask training code end ***************************************#  




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
lcd_init()
lcd_string("welcome ",LCD_LINE_1)
time.sleep(0.5)
lcd_byte(0x01,LCD_CMD) # 000001 Clear display
GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
GPIO.output(Red_led_pin,GPIO.LOW)  #LED OFF
GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF

while(1):
    lcd_byte(0x01,LCD_CMD) # 000001 Clear display
    lcd_string("Scan Your Face",LCD_LINE_1)
    #load the image from camera
    flag,img = capture.read()
    #check camera is working or not
    if flag :
        #this function return array for faces from image 
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            #draw rectangle over face
            #cv2.rectangle(img,(x,y),(w,h),(r,g,b),boarder_thickness)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,200),10)
            #slice the face fro image
            face = img[y:y+w,x:x+w,:]
            #resize the face into one size
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            pred = svm.predict(face)
            n = names[int(pred)]
            #print(n)
            temp_output = analogInput(0) # Reading from CH0
            temp_volts = Volts(temp_output)
            temp       = Temp(temp_output)
            print("{} deg C".format(temp))
            if(temp < 33):
                if((n == "Mask") ):
                    lcd_byte(0x01,LCD_CMD) # 000001 Clear display
                    lcd_string("Gate Open ",LCD_LINE_1)
                    GPIO.output(green_led_pin,GPIO.HIGH)  #LED ON
                    GPIO.output(Red_led_pin,GPIO.LOW)  #LED OFF
                    GPIO.output(dc_motor1,GPIO.LOW)  #Motor Open
                    GPIO.output(dc_motor2,GPIO.HIGH)  #MOTOR open
                    time.sleep(0.3)
                    GPIO.output(dc_motor1,GPIO.HIGH)  #Motor close
                    GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR close
                    time.sleep(0.3)
                    GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                    GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
                elif(n == "No Mask"):
                    lcd_string("Please wear mask",LCD_LINE_1)
                    lcd_byte(0x01,LCD_CMD) # 000001 Clear display
                    GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
                    GPIO.output(Red_led_pin,GPIO.HIGH)  #LED ON
                    GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                    GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
                    GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer On
                    time.sleep(0.5)
                    GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF
            else:
                lcd_string("High Body Temperature",LCD_LINE_1)
                lcd_byte(0x01,LCD_CMD) # 000001 Clear display
                GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
                GPIO.output(Red_led_pin,GPIO.HIGH)  #LED ON
                GPIO.output(dc_motor1,GPIO.LOW)  #Motor OFF
                GPIO.output(dc_motor2,GPIO.LOW)  #MOTOR OFF
                GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer On
                time.sleep(0.5)
                GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF
            cv2.putText(img,n,(x,y),font,1,(255,100,100),2) #show image
        cv2.imshow("output",img) #show image
        # 27 is ASCII value of escapse key
        #It is use to run thread which waiting for ESC button becuase of this image will show on window
        #if you not use this function then it will not load image
        if((cv2.waitKey(2) == 27)):
            break;
cv2.destroyAllWindows()
