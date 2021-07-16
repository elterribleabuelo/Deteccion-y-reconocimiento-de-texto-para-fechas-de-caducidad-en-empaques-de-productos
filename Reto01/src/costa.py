import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


costa = cv2.imread("Data/costa.png")
cv2.namedWindow("costa", cv2.WINDOW_NORMAL)
cv2.resizeWindow('costa', 16,20)
cv2.imshow("costa",costa)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Dimensiones originales

dim = costa.shape
print(dim)

# Relación de aspecto

ratio = dim[0] / dim[1]
print(ratio)

# Espacio de color RGB

costa = cv2.cvtColor(costa, cv2.COLOR_BGR2RGB)

# Hallamos el template

template = costa[370:780, 280:980]
plt.imshow(template)

# Escala de grises

costa_gray = cv2.cvtColor(costa, cv2.COLOR_RGB2GRAY)
template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)

# Template matching

res = cv2.matchTemplate(costa_gray, template_gray, cv2.TM_SQDIFF)
plt.imshow(res,cmap = "gray")
plt.show()
print(res.shape)

# Mínimos y maximos de "res"

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val, min_loc, max_loc)

# Rectángulo delimitador

x1, y1 = min_loc
x2, y2 = min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]
print(x1,y1,x2,y2)

# Dibujamos el rectángulo

costa_dib = costa.copy()
cv2.rectangle(costa_dib, (x1, y1), (x2, y2), (0, 255, 0), 3)
plt.imshow(costa_dib)

# =============================================================================
# Definimos el ROI
# =============================================================================

recorte = costa[y1:y2,x1:x2]
plt.figure(figsize=(16,20))
plt.imshow(recorte)

# Filtro gaussiano

recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
recorte_gray = cv2.GaussianBlur(recorte_gray,(5,5),0)
plt.imshow(recorte_gray,cmap = "gray")

# Dibujamos los bordes usando Canny

canny = cv2.Canny(recorte_gray,30,100)
plt.imshow(canny,cmap = "gray")

# Dibujamos los contornos

cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))

# Copia

copia_recorte = recorte.copy()

# Dibujamos los contornos y detectamos el rectángulo que contiene los datos
roi = []
for c in cnts:
  area = cv2.contourArea(c)
  x,y,w,h = cv2.boundingRect(c)
  epsilon = 0.01*cv2.arcLength(c,True)
  approx = cv2.approxPolyDP(c,epsilon,True)
  if len(approx) ==4 and 15000 <area <20000:
    print(area,len(approx))
    roi.append(approx[0][0])
    roi.append(approx[2][0])
    cv2.drawContours(copia_recorte, [c], 0, (0,255,0),2)

plt.imshow(copia_recorte)

# =============================================================================
# Delimitamos el ROI final
# =============================================================================

ocr = recorte[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
plt.imshow(ocr)
print(ocr.shape)

# Rotamos 
ocr = ndimage.rotate(ocr,270)
plt.imshow(ocr)

# Relación de aspecto
aspect_ratio = ocr.shape[0] / ocr.shape[1]

# Redimensionamos
ocr = cv2.resize(ocr,(300,int(250*aspect_ratio)),cv2.INTER_LINEAR)
plt.imshow(ocr)

# Guardamos la imagen OCR
cv2.imwrite('./ocr_costa.png',cv2.cvtColor(ocr,cv2.COLOR_BGR2RGB))

# Escala de grises
imagen_ocr = cv2.cvtColor(ocr,cv2.COLOR_RGB2GRAY)
plt.imshow(imagen_ocr,cmap= "gray")

# Filtro Gaussiano
imagen_ocr = cv2.GaussianBlur(imagen_ocr,(3,3),0)
plt.imshow(imagen_ocr,cmap = "gray")
plt.show()

# Umbralización
ocr = imagen_ocr.astype(np.uint8)
thresh1 = cv2.adaptiveThreshold(ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,15,6)

# Dilatacion
kernel = np.ones((1,1),np.uint8)
thresh1 = cv2.erode(thresh1,kernel,iterations = 1)

plt.imshow(thresh1,cmap = "gray")

# Aplicamos OCR
extractedInformation = pytesseract.image_to_string(thresh1, config= '--psm 06')
print(extractedInformation)

# Limpiamos algunos caracteres
extractedInformation = re.sub('\n', ' ', extractedInformation[:-1])
print(extractedInformation)

# Dibujamos la imagen con el OCR
costa_text = costa.copy()
cv2.putText(costa_text,extractedInformation,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
plt.figure(figsize=(16,20))
plt.imshow(costa_text)
plt.show()






