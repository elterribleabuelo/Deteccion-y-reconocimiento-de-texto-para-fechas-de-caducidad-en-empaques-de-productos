import numpy as np
import cv2
from sympy.solvers import solve

def ejes(rct):
    """
    params:
    rct: rectangulo delimitador de la línea roja del chocolate princesa
    return:
    p1 y p2: puntos que definen el eje principal sobre el cual se trazará las paralelas
    """
    #global pendiente
    x1,y1 = rct[0][0],rct[0][1]
    x2,y2 = rct[1][0],rct[1][1]
    x3,y3 = rct[2][0],rct[2][1]
    #x4,y4 = box[3][0],box[3][1]
    
    d1 = (x2-x1)**2 + (y2-y1)**2
    d2 = (x3-x2)**2 + (y3-y2)**2
    #d3 = (x4-x3)**2 + (y4-y3)**2
    #d4 = (x4-x1)**2 + (y4-y1)**2
    
    if (d2>d1):
        p1,p2 = [x2,y2],[x3,y3]
        pendiente = (y2-y3) / (x2-x3)
        if x2>x3:
            p1,p2 = p2,p1
            # p1 -> punto mas a la izquierda con respecto al eje X
            # p2 -> punto mas a la dercha con respecto al eje X
                            
    else:
        p1,p2 = [x2,y2],[x1,y1]
        pendiente = (y2-y1) / (x2-x1)
        if x2>x1:
            p1,p2 = p2,p1
            # p1 -> punto mas a la izquierda con respecto al eje X
            # p2 -> punto mas a la dercha con respecto al eje X
     
    return p1,p2,pendiente


def desplazar(p1,p2,dpz):
    """
    params:
    p1,p2: puntos que definen el eje principal
    dpz: Desplazamiento del eje principal hacia los ejes
    return:
    coord_1,coord_2: puntos que se encuentran arriba del eje principal
    coord_3,coord_4: puntos que se encuentran debajo del eje principal
    direccion: 0(en caso se desplaza en el eje X),
    1 (en caso se desplaze en el eje Y)
    """
    
    if(abs(p1[0]-p2[0])<20):
        coord_1,coord_2 = (p1[0] - dpz,p1[1]),(p2[0] - dpz,p2[1])
        coord_3,coord_4 = (p1[0] + dpz,p1[1]),(p2[0] + dpz,p2[1])
        direccion = 0  # Desplaza en el eje X
    else:
        coord_1,coord_2 = (p1[0],p1[1] - dpz),(p2[0],p2[1] - dpz)
        coord_3,coord_4 = (p1[0],p1[1] + dpz),(p2[0],p2[1] + dpz)
        direccion = 1 # Desplaza en el eje Y
    
    return coord_1,coord_2,coord_3,coord_4,direccion


def delimitar_zona(p1,p2,m,roi):
    """
    params:
    p1,p2: puntos que definen el eje principal
    m: pendiente de la recta que une p1,p2
    roi: imagen sobre la que se está analizando
    return:0
    """
    if m != 0:
     cv2.line(roi,(-1000,int((m)*-1000-m*p1[0]+p1[1])),
              (1000,int((m)*1000-m*p1[0]+p1[1])),
              (0,0,255),2)
     cv2.line(roi,(-1000,int((-1/m)*-1000-(-1/m)*p1[0]+p1[1])),
              (1000,int((-1/m)*1000-(-1/m)*p1[0]+p1[1])),
              (255,0,0),2)
     cv2.line(roi,(-1000,int((-1/m)*-1000-(-1/m)*p2[0]+p2[1])),
              (1000,int((-1/m)*1000-(-1/m)*p2[0]+p2[1])),
              (255,0,0),2)
     
     print(p1[0],p1[1],p2[0],p2[1],m)
     
    else:
        cv2.line(roi,(-1000,int((m)*-1000-m*p1[0]+p1[1])),
              (1000,int((m)*1000-m*p1[0]+p1[1])),
              (0,0,255),2)
        cv2.line(roi,(p1[0],p1[1]),
              (p1[0],0),
              (255,0,0),2)
        cv2.line(roi,(p1[0],0),
              (p1[0],1000),
              (255,0,0),2)
        cv2.line(roi,(p2[0],0),
              (p2[0],1000),
              (255,0,0),2)
        print(p1[0],p1[1],p2[0],p2[1],m)
    return 0


def hallar_ptos(coord1,coord2,m,dpz,direccion,exc,roi):
    """
    params:
    coord1,coord2: coordenadas del eje delimitador
    m: pendiente del eje principal
    dpz: desplazamiento del eje principal sobre cada uno de los ejes coordenados
    dirección: 0 (eje X) y 1 (eje Y)
    exc:
    roi: imagen en análisis
    return:
    -coord3: intersección de recta paralela a la recta principal( en direccion hacia abajo en el eje Y)
    y la recta perpendicular al eje principal que pasa por coord1.
    -coord4: intersección de recta paralela a la recta principal( en direccion hacia arriba en el eje Y)
    y la recta perpendicular a la recta principal que pasa por coord1.
    -coord5:intersección de recta paralela a la recta principal( en direccion hacia abajo en el eje Y)
    y la recta perpendicular al eje principal que pasa por coord2.
    -coord6:intersección de recta paralela a la recta principal( en direccion hacia arriba en el eje Y)
    y la recta perpendicular al eje principal que pasa por coord2.
    """
    x = Symbol('x')
    y = Symbol('y')
    
    # Dimensiones
    # Recta 1: Recta que une los puntos principales
    # y = coord1[1] + m_p(x - coord1[0])
    
    # Puntos resultantes : coord1 y coord2
    
    # Recta 2: Recta paralela a la recta principal con direccion y hacia abajo en el eje Y
    # y = coord1[1] + m_p(x - coord1[0]) - dpz
    
    # Recta 3: Recta paralela a la recta principal con direccion y hacia arriba en el eje Y
    # y = coord1[1] + m_p(x - coord1[0]) + dpz
    
    # Recta 4 : Recta perpendicular a la recta principal que pasa por coord1
    # y = coord1[1] - (1/m)*(x - coord1[0])
    
    # Intersección de recta 2 y 4
    if m !=0: # +exc: hacia abajo en el eje Y, 
        if direccion == 1: # Desplaza en el eje Y
            coord3 = solve([y-coord1[1]-m*(x-coord1[0]) + dpz, # recta 2
                            y-(coord1[1]+exc) + (1/m)*(x-(coord1[0]))], # recta 4
                           dict = True)
            
            coord3 = [int(coord3[0][x]),int(coord3[0][y])]
        
        else: # Desplaza en el eje X
           coord3 = solve([y-coord1[1]-m*(x-(coord1[0]-dpz)), # recta 2
                            y-(coord1[1]) + (1/m)*(x-(coord1[0]+exc))], # recta 4
                           dict = True)
            
           coord3 = [int(coord3[0][x]),int(coord3[0][y])] 
    
    else:
        if direccion == 1: # Desplaza en el eje Y
            coord3 =solve([y-coord1[1]-m*(x-coord1[0]) + dpz, # recta 2
                           x-coord1[0]], # recta 4
                          dict = True)
            
            coord3 = [int(coord3[0][x]),int(coord3[0][y])]
        else:
            coord3 =solve([y-coord1[1]-m*(x-(coord1[0]-dpz)), # recta 2
                           x-(coord1[0]+exc)], # recta 4
                          dict = True)
            
            coord3 = [int(coord3[0][x]),int(coord3[0][y])]
            
    # Punto resultante : coord3
    
    # Intersección de recta 3 y 4
    if m !=0:
        if direccion == 1:
            coord4 = solve([y-coord1[1]-m*(x-coord1[0]) - dpz, # recta3
                            y-(coord1[1]-exc) + (1/m)*(x-(coord1[0]))],# recta4
                           dict = True)
            
            coord4 = [int(coord4[0][x]),int(coord4[0][y])]
        else:
            coord4 = solve([y-coord1[1]-m*(x-(coord1[0] + dpz)),
                            y-(coord1[1]) + (1/m)*(x-(coord1[0]+exc))],
                           dict = True)
            
            coord4 = [int(coord4[0][x]),int(coord4[0][y])]
        
    else:
        if direccion == 1:
            coord4 = solve([y-coord1[1]-m*(x-coord1[0]) - dpz,
                            x-coord1[0]],
                           dict = True)
            coord4 = [int(coord4[0][x]),int(coord4[0][y])]
        else:
            coord3 =solve([y-coord1[1]-m*(x-(coord1[0] + dpz)),
                           x-(coord1[0]+exc)],
                          dict = True)
    
    # Punto resultante : coord4
    
    # Recta 5: Recta perpendicular a la recta principal que pasa por coord2
    # y = coord2[1] - (1/m)*(x - coord2[0])
    
    # Intersección de recta 2 y 5
    if m != 0:
        if direccion == 1:
            coord5 = solve([y-coord1[1]-m*(x-coord1[0]) + dpz,# recta 2
                            y-(coord2[1]-exc) + (1/m)*(x-(coord2[0]))], # recta 5
                           dict = True)
            
            coord5 = [int(coord5[0][x]),int(coord5[0][y])]
        
        else:
            coord5 = solve([y-coord1[1]-m*(x-(coord1[0] - dpz)),
                            y-(coord2[1]) + (1/m)*(x-(coord2[0]-exc))],
                           dict = True)
            
            coord5 = [int(coord5[0][x]),int(coord5[0][y])]
            
    
    else:
        if direccion == 1:
            coord5 = solve([y-coord1[1]-m*(x-coord1[0]) + dpz,
                            x-coord2[0]-exc],
                           dict = True)
            
            coord5 = [int(coord5[0][x]),int(coord5[0][y])]
        else:
            coord5 = solve([y-coord1[1]-m*(x-(coord1[0] - dpz)),
                            x-coord2[0]-exc],
                           dict = True)
            
            coord5 = [int(coord5[0][x]),int(coord5[0][y])]
            
    
    # Punto resultante : coord5
    
    # Intersección de recta 3 y 5
    if m != 0:
        if direccion == 1:
            coord6 = solve([y-coord1[1]-m*(x-coord1[0]) - dpz,
                            y-(coord2[1]) + (1/m)*(x-(coord2[0]-exc))],
                           dict = True)
            
            coord6 = [int(coord6[0][x]),int(coord6[0][y])]
        
        else:
            coord6 = solve([y-coord1[1]-m*(x-(coord1[0] + dpz)),
                            y-(coord2[1]) + (1/m)*(x-(coord2[0]-exc))],
                           dict = True)
            
            coord6 = [int(coord6[0][x]),int(coord6[0][y])]
            
            
    else:
        if direccion == 1:
            coord6 = solve([y-coord1[1]-m*(x-coord1[0]) - dpz,
                            x-coord2[0]-exc],
                           dict = True)
            
            coord6 = [int(coord6[0][x]),int(coord6[0][y])]
        
        else:
            coord6 = solve([y-coord1[1]-m*(x-(coord1[0] + dpz)),
                            x-coord2[0]-exc],
                           dict = True)
            
            coord6 = [int(coord6[0][x]),int(coord6[0][y])]
            
    return coord3,coord4,coord5,coord6


def tsf_perspectiva(p1,p2,p3,p4,roi,direccion):
    """
    params:
    p1,p2,p3,p4: puntos de referencia en sentido(supIzquierdo,supDerecho,infIzquierdo,infDerecho)
    roi: imagen en análisis
    direccion: eje X(0) o eje Y(1); si es eje X las rectas se desplazan en abcsisas, si es eje Y las rectas se desplazan
    en las ordenadas
    return:
    tsf: imagen corregida mediante transformación de perspectiva
    """
    max_x = max(p1[0],p2[0],p3[0],p4[0])
    min_x = min(p1[0],p2[0],p3[0],p4[0])
    max_y = max(p1[1],p2[1],p3[1],p4[1])
    min_y = min(p1[1],p2[1],p3[1],p4[1])
    
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    
    #print("Las dimensiones de la imagen transformada son:",dim_x,dim_y)
    if direccion == 1:
        pts1 = np.float32([[p1[0],p1[1]], # esquina superior izquierda
                           [p2[0],p2[1]], # esquina superior derecha
                           [p3[0],p3[1]], # esquina inferior izquierda
                           [p4[0],p4[1]]  # esquina inferior derecha
                           ])
        pts2 = np.float32([[0,0], [dim_x,0],
                           [0,dim_y], [dim_x,dim_y]
                           ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        tsf = cv2.warpPerspective(roi, M, (dim_x,dim_y))
        
        if tsf.shape[0] > tsf.shape[1]:
            tsf = ndimage.rotate(tsf,90)
    else:
        pts1 = np.float32([[p1[0],p1[1]], # esquina superior izquierda
                           [p3[0],p3[1]], # esquina superior derecha
                           [p2[0],p2[1]], # esquina inferior izquierda
                           [p4[0],p4[1]]  # esquina inferior derecha
                           ])
        pts2 = np.float32([[0,0], [dim_x,0],
                           [0,dim_y], [dim_x,dim_y]
                           ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        tsf = cv2.warpPerspective(roi, M, (dim_x,dim_y))
        if tsf.shape[0] > tsf.shape[1]:
            tsf = ndimage.rotate(tsf,90)
        
        # Region 1 --> Arriba
        # region1 = tsf_perspectiva(punto1, punto2, punto3, punto5, reg, direccion)
        # region1 = tsf_perspectiva(punto3, punto5, punto1, punto2, reg, direccion) --> Bien
        # p1,p2,p3,p4
        # Region 2 --> Abajo
        # region2 = tsf_perspectiva(punto1, punto2, punto4, punto6, reg, direccion)
        # region2 = tsf_perspectiva(punto1, punto2, punto4, punto6, reg, direccion) --> Bien
    
    return tsf