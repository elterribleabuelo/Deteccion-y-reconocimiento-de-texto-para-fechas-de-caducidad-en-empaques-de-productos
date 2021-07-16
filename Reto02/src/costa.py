import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from random import randrange
from sympy import integrate, init_printing
import pytesseract
from scipy import ndimage
import os
from skimage import color
from skimage.feature import corner_harris,corner_peaks
import utils
from imutils.object_detection import non_max_suppression
from paddleocr import PaddleOCR



cap = cv2.VideoCapture('../Data/videos/costa.avi')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilated = np.ones((5,5),np.uint8)
kernel_otsu = np.ones((7,7),np.uint8)
kernel_morpholigical = np.ones((1,1),np.uint8) #(3,3)
counter = 0
path_color = '../Data/costa/color'
path_grises = '../Data/costa/grises'
path_rotado = '../Data/costa/rotado'
path_aligment = '../Data/costa/alineado'
path_aligment_grises = '../Data/costa/alineado_grises'
path_roi = '../Data/costa/roi'
path_cuadrantes = '../Data/costa/cuadrantes'
path_inpaint = '../Data/costa/inpaint'
path_edge = '../Data/costa/edge'
path_txt = "../Data/costa/texto"
path_otsu = "../Data/costa/otsu"
path_multiplicacion = "../Data/costa/producto"
ocr = PaddleOCR(lang='es')



blancoBajo = np.array([0,0,200],np.uint8)
blancoAlto = np.array([18,255,255],np.uint8)
mask_laplaciano = np.array([[-1 , -1 , -1],[-1 , 9 , -1],[-1 , -1 , -1]])
mser = cv2.MSER_create(_delta = 6,
                       _min_area = 1500,
                       _max_area = 1550,
                       _max_variation = 0.75,
                       _min_diversity = 1.5)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

while True:
    ret,frame = cap.read()
    if ret == False:break
    dib_frame = imutils.resize(frame,width = 640)
    
    height = frame.shape[0]
    width =  frame.shape[1]
    
    
    fgmask = fgbg.apply(dib_frame)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask,kernel_dilated,iterations = 3)
    
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in cnts:
        if 50000 <cv2.contourArea(cnt)<60000:  # 45000- 60000
            ### Rectangulo rotado
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = box*2
            if (0<= box[0][0] <= width and 0<= box[1][0] <= width 
                and 0<= box[2][0] <= width and 0<= box[3][0] <= width ):
                
                ### Rectangulo ortogonal
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(dib_frame,(x,y),(x+w,y+h),(0,255,0),2)
                #roi = cv2.cvtColor(frame[2*y:2*(y+h),2*x:2*(x+w)],cv2.COLOR_BGR2RGB)
                #print(box)
                #print("Los puntos para la transformación de perspectiva son: " + 
                              #"{},{},{} y {} ".format(box[0], box[1],box[2],box[3]))
                puntos = utils.ord_pts(box)
                
                
                tsf = utils.transformacion(puntos,frame)
                if (tsf.shape[1]<tsf.shape[0]):
                    tsf = ndimage.rotate(tsf,90)
                
                    
                ### Eliminamos ruido
                dib_h = tsf.copy()
                dib_h = cv2.cvtColor(dib_h,cv2.COLOR_BGR2GRAY)
                dib_h = cv2.bitwise_not(dib_h)
                otsu = cv2.threshold(dib_h,  0,  255, cv2. THRESH_BINARY | cv2. THRESH_OTSU)[1]
                otsu = cv2.dilate(otsu,kernel_otsu,8)
                result = dib_h*otsu
                cv2.imwrite(os.path.join(path_otsu,str('OTSU') + str(counter) + str('.png')),
                            otsu)
                cv2.imwrite(os.path.join(path_multiplicacion,str('MULT') + str(counter) + str('.png')),
                            result)
                pintar = tsf.copy()
                coordinates, bboxes = mser.detectRegions(result)
                for bbox in bboxes:
                    x, y, w, h = bbox
                    if (1.0<h/w<2.0 and 60<h<120):
                        #aspect_ratio = h / w
                        #txt = cv2.resize(tsf[y-30:y+h+30,x-50:x+w+50],(300,int(250*aspect_ratio)),cv2.INTER_LINEAR)
                        #imagen_ocr = cv2.GaussianBlur(txt,(3,3),0)
                        ### OCR ###
                        #extractedInformation = ocr.ocr(imagen_ocr, det = True,rec = True, cls = True)
                        # print(extractedInformation)
                        cv2.rectangle(pintar, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(pintar, (x-50, y-30), (x + w + 50, y + h + 30), (255, 255, 0), 2)
                        cv2.imwrite(os.path.join(path_txt,str('MSER') + str(counter) + str('.png')),
                                pintar)
                        cv2.imwrite(os.path.join(path_roi,str('ROI') + str(counter) + str('.png')),
                                tsf[y-30:y+h+30,x-50:x+w+50])
                cv2.imwrite(os.path.join(path_cuadrantes,str('Afilado') + str(counter) + str('.png')),
                            dib_h)
                
                # Dibujamos los contornos y detectamos el rectángulo que contiene los datos
                cv2.imwrite(os.path.join(path_aligment_grises,str('Correcto_gris') + str(counter) + str('.png')),
                                    cv2.cvtColor(tsf,cv2.COLOR_BGR2GRAY))
                cv2.imwrite(os.path.join(path_aligment,str('Correcto') + str(counter) + str('.png')),
                                    tsf)
                cv2.imwrite(os.path.join(path_rotado,str('Rotacion') + str(counter) + str('.png')),
                                    frame)
                counter = counter + 1
                
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame', dib_frame)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    elif k == 32:
        cv2.waitKey()

cap.release()
cv2.destroyAllWindows()

#https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x/18614498
#https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html
#https://stackoverflow.com/questions/47595684/extract-mser-detected-areas-python-opencv


