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


def elegir_linea(lines,roi,cnt):
    """
    params:
    lines: lineas de Hough
    roi: región donde se dibuja la línea de Hough
    cnt: cantidad de líneas de Hough encontradas
    
    return:
    x1,y1: par ordenado 1 de puntos que pertenecen a la recta
    x2,y2: par ordenado 2 de puntos que pertenecen a la recta
    a: coseno del angulo formado por la línea de Hough y el eje X
    b: seno del angulo formado por la línea de Hough y el eje X
    """
    # Obtener los valores de rho (distacia)
    rho = lines[cnt][0][0]
     
    # y de theta (ángulo)
    theta = lines[cnt][0][1]
     
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
    
    return x1,y1,x2,y2,a,b


def calc_area(a,b,y1,y2,x1,x2,dpz,roi):
     """
     params:
     a: coseno del angulo formado por la línea de Hough y el eje X
     b: seno del angulo formado por la línea de Hough y el eje X
     x1,y1: par ordenado 1 de puntos que pertenecen a la recta
     x2,y2: par ordenado 2 de puntos que pertenecen a la recta
     dpz: desplazamiento de la recta de Hough en el eje X
     roi: región en la cual se halla la recta de Hough
     
     return:
     1: la recta paralela a Hough se desplaza en el eje Y dirección negativa
     0: la recta paralela a Hough se desplaza en el eje Y dirección positiva
    """
     # Ecuacion de la recta 
     # Pendiente de la recta
     m = -(a/b)
     # Ecuación cartesiana
     f = m*x + (y1-m*x1)
     # Area entre f y y = 0
     A1 = integrate(f-0, (x,0,roi.shape[1]))
     print("Area 1 =",A1)
     # Area entre f y y = roi.shape[1]
     A2 = roi.shape[0]*roi.shape[1]-A1
     print("Area 2 =",A2)
     if A1 > A2:
        print("mover dirección hacia ")
        cv2.line(roi,(x1,y1-dpz),(x2,y2-dpz),(255,0,0),2) # Recta paralea a Hough
        return 1
     elif A1 < A2:
        print("mover dirección contrario")
        cv2.line(roi,(x1,y1+dpz),(x2,y2+dpz),(255,0,0),2) # Recta paralea a Hough
        return 0

def hallar_ptos(orientacion,a,b,y1,x1,dpz,roi,limit_inferior,limit_superior):
    """
    params:
    a: coseno del angulo formado por la recta de Hough y el eje X
    b: seno del angulo formado por la recta de Hough y el eje Y
    x1,y1: coordenadas por donde pasa la recta de Hough
    dpz: Desplazamiento de la recta de Hough que permite hallar la otra línea
    roi: Región de interés sobre la que se aplica la función
    limit_superior: porcentaje menor del ancho de la imagen en la cual se traza la paralela al eje Y
    limit_inferior: porcentaje mayor del ancho de la imagen en la cual se traza la paralela al eje Y
    
    return:
    p1,p2,p3,p4: puntos de corte de la recta de Hough y la recta paralela
    a esta con la recta x = limit_inferior o x = limit_superior
    """
    xi = int(limit_inferior*roi.shape[1])
    xf = int(limit_superior*roi.shape[1])
    m = -(a/b)
    
    # Punto 1 --> (0,yn) --> (limit_inferior,yn)
    yn = int(y1 + m*(xi - x1))
    p1 = (xi,yn)
    
    # Punto 2 --> (roi.shape[1],p2_y)--> (limit_superior,yn)
    p2_y = int(y1 + m*(xf-x1))
    p2 = (xf,p2_y)
    
    if orientacion == 1:
        # Los puntos 3 y 4 son solo desplazamientos hacia arriba (-)
        p3 = (xi, yn-dpz)
        p4 = (xf,p2_y-dpz)
        # Dibujamos puntos en la imagen
        cv2.circle(roi,p1,5,(255,255,0),2) 
        cv2.circle(roi,p2,5,(255,255,0),2) 
        cv2.circle(roi,p3,5,(255,255,0),2) 
        cv2.circle(roi,p4,5,(255,255,0),2)
    
    elif orientacion == 0:
        # Los puntos 3 y 4 son solo desplazamientos hacia abajo (+)
        p3 = (xi, yn+dpz)
        p4 = (xf,p2_y+dpz)
        cv2.circle(roi,p1,5,(255,255,0),2) 
        cv2.circle(roi,p2,5,(255,255,0),2) 
        cv2.circle(roi,p3,5,(255,255,0),2) 
        cv2.circle(roi,p4,5,(255,255,0),2)
    
    return p1,p2,p3,p4


def ordenar_puntos(p1,p2,p3,p4):
    """
    params:
    p1,p2,p3,p4: puntos que forman la region de interés
    
    return:
    
    n_puntos[0],n_puntos[1],n_puntos[2],n_puntos[3]: puntos ordenados
    en el orden: (superior_izquierdo,superior_derecho,inferior_izquierdo,
                  inferior_derecho)
    """
    puntos = []
    puntos.insert(0,[p1[0],p1[1]]) # (0,yn)  --> n_puntos[0]
    puntos.insert(1,[p3[0],p3[1]]) # (0,yn+-30) --> n_puntos[1]
    puntos.insert(2,[p2[0],p2[1]]) # (ancho,yn) --> n_puntos[2]
    puntos.insert(3,[p4[0],p4[1]]) # (ancho,yn+-30) -->n_puntos[3]
    #print(puntos[0])
    
    n_puntos = np.concatenate([[puntos[0]], # (0,yn)  --> n_puntos[0]
                               [puntos[2]], # (ancho,yn) --> n_puntos[1]
                               [puntos[1]], # (0,yn+-30) --> n_puntos[2]
                               [puntos[3]   # (ancho,yn+-30) -->n_puntos[3]
                              ]]).tolist()
    print("n_puntos=",n_puntos)
    
    if (n_puntos[0][1] < n_puntos[2][1]):
        # yn < yn + 30
        # Recta se desplaza hacia y= roi.shape[1]
        return n_puntos[0],n_puntos[1],n_puntos[2],n_puntos[3]
    
    else:
        # Recta se despalza hacia y = 0
        return n_puntos[2],n_puntos[3],n_puntos[0],n_puntos[1]

def tsf_perspectiva(p1,p2,p3,p4,roi):
    """
    params:
    p1,p2,p3,p4: puntos que conforman la region (ordenados anteriormente)
    roi: region que encierra los cuatro puntos
        
    return:
    tsf: transformación de perspectiva en base a p1,p2,p3 y p4
    """
    max_x = max(p1[0],p2[0],p3[0],p4[0])
    min_x = min(p1[0],p2[0],p3[0],p4[0])
    max_y = max(p1[1],p2[1],p3[1],p4[1])
    min_y = min(p1[1],p2[1],p3[1],p4[1])
    
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    
    print("Las dimensiones de la imagen transformada son:",dim_x,dim_y)  
    pts1 = np.float32([[p1[0],p1[1]], [p2[0],p2[1]], [p3[0], p3[1]], [p4[0],p4[1]]])
    pts2 = np.float32([[0,0], [dim_x,0],
                       [0,dim_y], [dim_x,dim_y]
                       ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    tsf = cv2.warpPerspective(roi, M, (dim_x,dim_y))
    
    return tsf

def recognition(tsf):
    ocr = tsf.copy()
    ocr = ndimage.rotate(ocr,180)
    
    return ocr


def ord_pts(puntos):
    n_puntos = np.concatenate([[puntos[0]],[puntos[1]],[puntos[2]],[puntos[3]]]).tolist()
    #print("n_puntos",n_puntos)
    
    # ordenamos con respecto al eje Y
    y_order = sorted(n_puntos,key = lambda n_puntos:n_puntos[1])
    #print("y_order:",y_order)
    # Ordenamos los puntos que se encuebtran mas hacia arriba 
    # con respecto al eje X
    # x1_order: contiene los vertices superiores(izq,der)
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key = lambda x1_order:x1_order[0])
    #print("x1_order",x1_order)
    # Ordenamos los puntos que se encuebtran mas hacia abajo 
    # con respecto al eje X
    # x2_order: contiene los vertices inferiores(izq,der)
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key = lambda x2_order:x2_order[0])
    #print("x2_order",x2_order)
    
    return [x1_order[0],x1_order[1],x2_order[0],x2_order[1]]


def transformacion(puntos,roi):
    max_x = max(puntos[0][0],puntos[1][0],puntos[2][0],puntos[3][0])
    min_x = min(puntos[0][0],puntos[1][0],puntos[2][0],puntos[3][0])
    max_y = max(puntos[0][1],puntos[1][1],puntos[2][1],puntos[3][1])
    min_y = min(puntos[0][1],puntos[1][1],puntos[2][1],puntos[3][1])
    
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    
    print("Las dimensiones de la imagen transformada son:",dim_x,dim_y)  
    pts1 = np.float32([[puntos[0]], [puntos[1]], [puntos[2]], [puntos[3]]])
    pts2 = np.float32([[0,0], [dim_x,0],
                       [0,dim_y], [dim_x,dim_y]
                       ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    tsf = cv2.warpPerspective(roi, M, (dim_x,dim_y))
    
    return tsf


