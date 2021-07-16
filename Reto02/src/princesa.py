import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from random import randrange
from sympy import integrate, init_printing
import pytesseract
from scipy import ndimage
import os
import utils
from sympy.solvers import solve
from sympy import Symbol
from paddleocr import PaddleOCR
import recognition
import transformaciones

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilated = np.ones((5,5),np.uint8)
kernel_morpholigical = np.ones((1,1),np.uint8)
counter = 0
path = '../Data/princesa/'
path_canny = '../Data/princesa/canny'
path_color = '../Data/princesa/color'
path_alineado = '../Data/princesa/alineado'
path_roi = '../Data/princesa/roi'
path_roi2 = '../Data/princesa/roi2'

ocr = PaddleOCR(lang='es')
cap = cv2.VideoCapture('../Data/videos/princesa.avi')
  
while (True):
    
  ret, frame = cap.read()
  
  if ret == False: break
  dib_frame = imutils.resize(frame,width = 640)
  gray = cv2.cvtColor(dib_frame,cv2.COLOR_BGR2GRAY)
  
  # Especificamos los puntos extremos del area a analizar
  area_pts = np.array([[120,30],
                        [dib_frame.shape[1],30],
                        [dib_frame.shape[1],dib_frame.shape[0]],
                        [120, dib_frame.shape[0]]
                        ])
                        
                        
                        
  # Con ayuda de una imagen auxiliar, determinamos el area
  # sobre el cual actuara el detector de movimiento
  imAux = np.zeros(shape=(dib_frame.shape[:2]), dtype= np.uint8)
  imAux = cv2.drawContours(imAux,[area_pts],-1,(255,0,0),-1)
  
  # Imagen sobre el cual se analizara substraccion de fondo
  image_area = cv2.bitwise_and(gray,gray,mask = imAux)
  
  dibujar = dib_frame.copy()
  
  # Substracción de fondo
  fgmask = fgbg.apply(image_area)
  
  # Operaciones morfologicas
  # fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)
  
  fgmask = cv2.dilate(fgmask,kernel_dilated,iterations = 8)
  
  # Encontramos los contornos presentes en fgmask , para luego
  # basandonos en su area poder determinar si existe movimiento
  
  cnts = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
  cv2.drawContours(dibujar,[area_pts],-1,(255,0,0),2)
  
  # Encontramos los contornos presentes de fgmask , para luego basándonos
  # en su area poder determinar si existe movimiento
  
  for cnt in cnts:
        if cv2.contourArea(cnt) > 30000:
            
            cx,cy,cw,ch = cv2.boundingRect(cnt)
            cv2.rectangle(dibujar,(cx,cy),(cx+cw,cy+ch),(0,255,0),2)
            
            roi = cv2.cvtColor(frame[2*cy:2*(cy+ch),2*cx:2*(cx+cw)],cv2.COLOR_BGR2RGB)
            reg = roi.copy()
            ancho = roi.shape[1]
            alto = roi.shape[0]
            recorte = roi.copy()
            
            # Filtros
            recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            recorte_gray = cv2.GaussianBlur(recorte_gray,(3,3),0)
            
            # Canny
            canny = cv2.Canny(recorte_gray,40,120) # 40-120
            
            # Contornos
            cnts_canny,_ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            
            counter = counter + 1
            
            for c in cnts_canny:
                area = cv2.contourArea(c)            
                epsilon = 0.008*cv2.arcLength(c,True) # calculamos perimetro del contorno
                approx = cv2.approxPolyDP(c,epsilon,True) # aproximar la forma
                if len(approx) == 4 and area > 800:
                  rect = cv2.minAreaRect(c)
                  box = cv2.boxPoints(rect)
                  box = np.int0(box)
            
                  punto1,punto2,m = transformaciones.ejes(box)  
                  coord_1,coord_2,coord_3,coord_4,direccion = transformaciones.desplazar(punto1,punto2,80) # puntos de desplazamiento de la recta principal
                  punto3,punto4,punto5,punto6 = transformaciones.hallar_ptos(punto1,punto2,m,80,direccion,0,roi)
                  
                  # Region 1
                  region1 = transformaciones.tsf_perspectiva(punto3, punto5, punto1, punto2, reg, direccion)
        
                  # Region 2
                  region2 = transformaciones.tsf_perspectiva(punto1, punto2, punto4, punto6, reg, direccion)
                  
                  desc1 = recognition.ocr(region1.copy(),ocr)
                  desc2 = recognition.ocr(region2.copy(),ocr)
                  
                  print("OCR1:",desc1)
                  print("OCR2:",desc2)
                  
                  
                  cv2.drawContours(roi,[box],0,(0,0,255),2)
                  cv2.line(roi,coord_1,coord_2,(0,255,0),2)
                  cv2.line(roi,coord_3,coord_4,(0,255,0),2)
                  transformaciones.delimitar_zona(punto1,punto2,m,roi)
                  cv2.imwrite(os.path.join(path_color,str('imagen') + str(counter) + str('.jpg')),
                            cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
                  cv2.imwrite(os.path.join(path_canny,str('canny') + str(counter) + str('.jpg')),
                            canny)
                  cv2.imwrite(os.path.join(path_roi,str('region1_') + str(counter) + str('.jpg')),
                            cv2.cvtColor(region1,cv2.COLOR_BGR2RGB))
                  cv2.imwrite(os.path.join(path_roi,str('region2_') + str(counter) + str('.jpg')),
                            cv2.cvtColor(region2,cv2.COLOR_BGR2RGB))
  
  cv2.imshow('video', dibujar)
  cv2.imshow('mask', fgmask)
  
  k = cv2.waitKey(40) & 0xFF
  
  if k == 27:
      break
  elif k == 32:
    cv2.waitKey()

cap.release()
cv2.destroyAllWindows()