import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pytesseract
from pytesseract import Output
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

sal = cv2.imread("sal.png")

dim = sal.shape

ratio = dim[0] / dim[1]

sal = cv2.cvtColor(sal, cv2.COLOR_BGR2RGB)

recorte = sal.copy()

recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
recorte_gray = cv2.GaussianBlur(recorte_gray,(3,3),0)

canny = cv2.Canny(recorte_gray,80,170)

kernel = np.ones((3,3),np.uint8)
canny_final = cv2.dilate(canny,kernel,iterations = 1)

cnts,_ = cv2.findContours(canny_final,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

copia_recorte = sal.copy()
roi = []

for c in cnts:
  area = cv2.contourArea(c)
  x,y,w,h = cv2.boundingRect(c)
  epsilon = 0.01*cv2.arcLength(c,True)
  approx = cv2.approxPolyDP(c,epsilon,True)
  if len(approx) == 4 and 50000 < area < 60000 :
    print(area,len(approx))
    roi.append(approx[0][0])
    roi.append(approx[2][0])
    cv2.drawContours(copia_recorte, [c], 0, (0,255,0),2)

x1 = roi[0][1]
x2 = roi[0][0]
y1 = roi[1][0]
y2 = roi[1][1]

ocr = ndimage.rotate(ocr,180)
plt.figure(figsize=(16,20))
plt.axis('off')
plt.imshow(ocr)

ocr = cv2.cvtColor(ocr,cv2.COLOR_RGB2GRAY)

ocr = cv2.GaussianBlur(ocr,(1,1),0)

thresh1 = cv2.adaptiveThreshold(ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 5)

extractedInformation = pytesseract.image_to_string(thresh1, config= '--psm 06')

extractedInformation = list(extractedInformation)

final = []
for i in range(len(extractedInformation)):
  if (extractedInformation[i].isupper() == True or 
      extractedInformation[i] == '0' or
      extractedInformation[i] == '1' or
      extractedInformation[i] == '2' or
      extractedInformation[i] == '3' or
      extractedInformation[i] == '4' or
      extractedInformation[i] == '5' or
      extractedInformation[i] == '6' or
      extractedInformation[i] == '7' or
      extractedInformation[i] == '8' or
      extractedInformation[i] == '9' or
      extractedInformation[i] == '.' or
      extractedInformation[i] == ' ' or
      extractedInformation[i] == ':' or
      extractedInformation[i] == '_'):
    final.append(extractedInformation[i])
    
final = ''.join(final)
sal_text = sal.copy()
cv2.putText(sal_text,final,(400,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.imwrite('resultado_sal.png',cv2.cvtColor(sal_text,cv2.COLOR_BGR2RGB))