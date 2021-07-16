import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pytesseract
from pytesseract import Output
import re
import os



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path = "../Data/"

princesa = cv2.imread("../Data/princesa.png")
cv2.namedWindow("princesa", cv2.WINDOW_NORMAL)
cv2.imshow("princesa",princesa)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Espacio de color RGB
princesa = cv2.cvtColor(princesa, cv2.COLOR_BGR2RGB)

# Hallamos el template
template = princesa[480:700,250:920]
plt.imshow(template)


# Escala de grises
princesa_gray = cv2.cvtColor(princesa, cv2.COLOR_RGB2GRAY)
template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)


# Template matching
res = cv2.matchTemplate(princesa_gray, template_gray, cv2.TM_SQDIFF)
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
princesa_dib = princesa.copy()
cv2.rectangle(princesa_dib, (x1, y1), (x2, y2), (0, 255, 0), 3)
plt.imshow(princesa_dib)


# =============================================================================
# Definimos el ROI
# =============================================================================

recorte = princesa[y1:y2,x1:x2]
plt.figure(figsize=(16,20))
plt.imshow(recorte)
print(recorte.shape)


# Filtro gaussiano

recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
recorte_gray = cv2.GaussianBlur(recorte_gray,(5,5),0)
plt.imshow(recorte_gray,cmap = "gray")

# Dibujamos los bordes usando Canny
canny = cv2.Canny(recorte_gray,30,210)
cv2.imshow('grises',recorte_gray)
cv2.imshow('bordes',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
# Usamos Transformada de Hough
# =============================================================================

lines = cv2.HoughLines(canny,1,np.pi/180,200,None)
print("Hough")
print(lines)
print(lines.shape)
print(type(lines))
#print(lines[0][0][0])
#print(lines[0][0][1])
#print(lines[1][0][0])
#print(lines[1][0][1])

dibujar = recorte.copy()

if lines is not None:
    # Recorrer los resultados
    for i in range(0, len(lines)):
        
        # Obtener los valores de rho (distacia)
        rho = lines[i][0][0]
        
		# y de theta (ángulo)
        theta = lines[i][0][1]
        
		# guardar el valor del cos(theta)
        a = np.cos(theta)
        
		# guardar el valor del sen(theta)
        b = np.sin(theta)
        
		# guardar el valor de r cos(theta)
        x0 = a*rho
        
		# guardar el valor de r sen(theta), todo se está haciendo de forma paramétrica
        y0 = b*rho
        
		# Ahora todo se recorrerá de -1000 a 1000 pixeles
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
		# Mostrar los valores hallados
        print(x0,y0)
        print("({},{})  ({},{})".format(x1,y1, x2,y2))
		# Generar las líneas para montarlas en la imagen original
        cv2.line(dibujar,(x1,y1),(x2,y2),(0,0,255),2) # Recta de Hough
        cv2.circle(dibujar,(int(x0),int(y0)),5,(255,255,0),1) # circulo en x1,y1
        cv2.circle(dibujar,(x1,y1),5,(255,255,0),3) # circulo en x1,y1
        cv2.circle(dibujar,(x2,y2),5,(255,255,0),3) # circulo en x2,y2
        cv2.circle(dibujar,(x1,y1+35),5,(255,255,0),3) # circulo en x2,y2
        cv2.circle(dibujar,(x2,y2+35),5,(255,255,0),3) # circulo en x2,y2
        cv2.line(dibujar,(x1,y1+35),(x2,y2+35),(0,255,0),2) # Recta paralea a Hough

print(x0,y0)
roi = recorte[int(y0-30):int(y0+20),int(x0):int(x0+600)]
# Mostrar la imagen original con todas las líneas halladas
cv2.imshow('bordes',canny)
cv2.imshow('Hough', dibujar)
cv2.imshow('roi',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
# Definimos la ecuacion de la recta en forma cartesiana(pendiente)
# Calculamos los interceptos con el eje X e Y
# =============================================================================

# Pendiente de la recta de Hough

m = -(a/b)

print("La pendiente de la recta de Hough es : ", m)

# Intercepto con el eje Y --> yn

yn = int(y0 - m*x0)

print("El intercepto de la recta de Hough con el eje Y es : ", yn)

# Intercepto con el eje X--> xn

xn = int(x0-(y0/m))

print("El intercepto de la recta de Hough con el eje X es : ", xn)

# =============================================================================
# Delimitamos ROI
# =============================================================================

roi = recorte[int(yn-30):int(yn+40),0:recorte.shape[1]]
recorte.shape
cv2.imshow('roi',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(roi)

# =============================================================================
# Hallamos los 4 puntos de interes
# =============================================================================

# Punto 1 --> (0,yn)

p1 = (0,yn)
print(p1)

# Punto 2 -->(recorte.shape[1],p2_y)

p2_y = int(y0 + m*(recorte.shape[1]-x0))
p2 = (recorte.shape[1],p2_y)
print(p2)


# Punto 3 --> (0,yn+35)

p3 = (0, yn + 35)
print(p3)

# Punto 4 -->(recorte.shape[2],p4_y)

p4_y = int(yn + 35 +m*(recorte.shape[1]-0))
p4 = (recorte.shape[1],p4_y)
print(p4)


print("Los puntos para la transformación de perspectiva son: " + 
      "{},{},{} y {} ".format(p1, p2, p3, p4))

# =============================================================================
# Visualización de los puntos de referencia
# =============================================================================

pt_ref = recorte.copy()

cv2.circle(pt_ref,p1,5,(255,255,0),2) 
cv2.circle(pt_ref,p2,5,(255,255,0),2) 
cv2.circle(pt_ref,p3,5,(255,255,0),2) 
cv2.circle(pt_ref,p4,5,(255,255,0),2) 

cv2.imshow('Puntos de referencia',cv2.cvtColor(pt_ref,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Transformación de perspectiva
# =============================================================================

max_x = max(p1[0],p2[0],p3[0],p4[0])
min_x = min(p1[0],p2[0],p3[0],p4[0])
max_y = max(p1[1],p2[1],p3[1],p4[1])
min_y = min(p1[1],p2[1],p3[1],p4[1])

dim_x = max_x - min_x
dim_y = max_y - min_y

print("Las dimensiones de la imagen transformada son:",dim_x,dim_y)
#### Coodenadas de los vertices de la zona de interes
#### en sentido horario(sup-izq ->sup-der -> inf-izq -> inf-der)    
pts1 = np.float32([[p1[0],p1[1]], [p2[0],p2[1]], [p3[0], p3[1]], [p4[0],p4[1]]])

#### Array que corresponde a la nueva imagen
pts2 = np.float32([[0,0], [dim_x,0],
                   [0,dim_y], [dim_x,dim_y]
                   ])

### cv2.getPerspectiveTransform: funcion que calcula la transformacion
### de perspectiva a partir de 4 puntos, por lo tanto, de esta se obtendra
### una matriz de 3 x 3 ()
M = cv2.getPerspectiveTransform(pts1, pts2)
tsf = cv2.warpPerspective(recorte, M, (dim_x,dim_y)) # debe corresponder a pts2


cv2.imshow('Imagen', recorte)
cv2.imshow('Tranformacion perspectiva', tsf)
cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
# Aplicamos OCR a la imagen contenida en la variable tsf
# =============================================================================

ocr = tsf.copy()
ocr = cv2.cvtColor(ocr, cv2.COLOR_BGR2GRAY)
ocr = cv2.GaussianBlur(ocr,(3,3),0)

cv2.imshow('OCR', ocr)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh1 = cv2.adaptiveThreshold(ocr, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 13, 8)

# cv2.ADAPTIVE_THRESH_GAUSSIAN_C y (13,8) --> Da buenos resultados
thresh1 = ndimage.rotate(thresh1,180)

cv2.imshow('Umbralizacion', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(pytesseract.image_to_osd(tsf, output_type=Output.DICT))
extractedInformation = pytesseract.image_to_string(thresh1, config= '--psm 06')
cv2.putText(princesa,extractedInformation,(100,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
cv2.imwrite(os.path.join(path,str('imagen') + str('.jpg')),cv2.cvtColor(princesa,cv2.COLOR_BGR2RGB))
print(extractedInformation)


#ay fA "12901074 260921 So |





