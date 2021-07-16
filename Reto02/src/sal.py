import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from random import randrange
from sympy import integrate, init_printing
from sympy.abc import x
import pytesseract
from scipy import ndimage
import os
import utils
import east
import keras_ocr
import recognition
from imutils.object_detection import non_max_suppression
from imutils.contours import sort_contours
from easyocr import Reader


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilated = np.ones((5,5),np.uint8)
kernel_morpholigical = np.ones((1,1),np.uint8) #(3,3)
counter = 0
path = '../Data/sal/color'
path_canny = '../Data/sal/Canny'
path_dilate = '../Data/sal/Canny_dilatacion'
path_hough = '../Data/sal/Hough'
path_ocr = '../Data/sal/ocr'
path_equalizate = '../Data/sal/ocr_equalizado'
path_text = '../Data/sal/contornos'
path_digitos ='../Data/sal/digitos'
path_roi = '../Data/sal/roi_final'
custom_config = r'--oem 3 --psm 6 outputbase digits'


cap = cv2.VideoCapture('../Data/videos/sal.avi')
reader = Reader(['en'])

while (True):
    
  ret, frame = cap.read()
  
  if ret == False: break
  frame = imutils.resize(frame,width = 640)
  
  dibujar = frame.copy()
  
  # Substracción de fondo
  fgmask = fgbg.apply(frame)

  
  fgmask = cv2.dilate(fgmask,kernel_dilated,iterations = 8)
  
  # Encontramos los contornos presentes en fgmask , para luego
  # basandonos en su area poder determinar si existe movimiento
  
  cnts = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
  
  for cnt in cnts:
        if cv2.contourArea(cnt) > 180000:
            
            cx,cy,cw,ch = cv2.boundingRect(cnt)
            cv2.rectangle(dibujar,(cx,cy),(cx+cw,cy+ch),(0,255,0),2)
            
            # Delimitamos el ROI
            roi = cv2.cvtColor(frame[cy:cy+ch,cx:cx+cw],cv2.COLOR_BGR2RGB)
            counter = counter + 1
            
            # Hacemos una copia
            recorte = roi.copy()
            
            # Filtros
            recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            recorte_gray = cv2.GaussianBlur(recorte_gray,(3,3),0)
            
            # Canny
            canny = cv2.Canny(recorte_gray,80,200)
            
            # Erocion
            canny_final = cv2.erode(canny,kernel_morpholigical,iterations = 1)
            
            # Transformada de Hough
            lines = cv2.HoughLines(canny_final,1,np.pi/180,140,None) # 135
            
            # Dibujar Hough
            if lines is not None:
                num_lineas = lines.shape[0] # numero de lineas de Hough
                
                if (num_lineas == 1):
                    cnt = num_lineas -1 
                    x1,y1,x2,y2,a,b = utils.elegir_linea(lines,roi,cnt)
                    mst_ocr = roi.copy()
                    
                    # Generar las líneas para montarlas en la imagen original
                    cv2.line(roi,(x1,y1),(x2,y2),(0,255,0),1) # Recta de Hough
                    orientacion = utils.calc_area(a,b,y1,y2,x1,x2,85,roi)
                    p1,p2,p3,p4 = utils.hallar_ptos(orientacion,a,b,y1,x1,85,roi,0.15,0.80)
                    p1,p2,p3,p4 = utils.ordenar_puntos(p1,p2,p3,p4)
                    tsf = utils.tsf_perspectiva(p1,p2,p3,p4,mst_ocr)
                    ocr = utils.recognition(tsf)
                    ocr = cv2.cvtColor(ocr,cv2.COLOR_BGR2RGB)
                    
                    txt = utils.recognition(tsf)
                    
                    #### OCR ####
                    result = reader.readtext(ocr)
                    long = len(result)
                    textos = []
                    for k in range(long):
                        textos.append(result[k][1])
                    desc = ' '.join(textos)
                    
                  
                    cv2.putText(dibujar,desc,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                    cv2.imwrite(os.path.join(path_ocr,str('ocr') +  str(counter) + str('.jpg')),ocr)
                    
                else:
                    
                    cnt = randrange(num_lineas) -1
                    x1,y1,x2,y2,a,b = utils.elegir_linea(lines,roi,cnt)
                    mst_ocr = roi.copy()
                    # Generar las líneas para montarlas en la imagen original
                    cv2.line(roi,(x1,y1),(x2,y2),(0,255,0),1) # Recta de Hough
                    orientacion =  utils.calc_area(a,b,y1,y2,x1,x2,95,roi)
                    p1,p2,p3,p4 = utils.hallar_ptos(orientacion,a,b,y1,x1,95,roi,0.15,0.80)
                    p1,p2,p3,p4 = utils.ordenar_puntos(p1,p2,p3,p4)
                    tsf = utils.tsf_perspectiva(p1,p2,p3,p4,mst_ocr)
                    ocr = utils.recognition(tsf)
                    ocr = cv2.cvtColor(ocr,cv2.COLOR_BGR2RGB)
                    
                    txt = utils.recognition(tsf)
                    
                    #### OCR ####
                    result = reader.readtext(ocr)
                    long = len(result)
                    textos = []
                    for k in range(long):
                        textos.append(result[k][1])
                    desc = ' '.join(textos)
             
                    cv2.putText(dibujar,desc,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                    cv2.imwrite(os.path.join(path_ocr,str('ocr') +  str(counter) + str('.jpg')),ocr)
            cv2.imwrite(os.path.join(path,str('imagen') + str(counter) + str('.jpg')),cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(path_canny,str('canny') + str(counter) + str('.jpg')),canny)
            cv2.imwrite(os.path.join(path_dilate,str('canny_final') + str(counter) + str('.jpg')),canny_final)
  
  cv2.imshow('video', dibujar)
  cv2.imshow('mask', fgmask)
  
  k = cv2.waitKey(80) & 0xFF
  if k == 27:
      break
  elif k == 32:
    cv2.waitKey()

cap.release()
cv2.destroyAllWindows()


















