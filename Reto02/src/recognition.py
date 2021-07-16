import numpy as np
import keras_ocr
import cv2

def ocr(image,ocr):
    """
    params:
    image: imagen en formato np.array donde se encuentra
    la imagen donde se quiere aplicar OCR
    
    return:
    predictions: texto reoconocido en la imagen
    """
    textos = []
    gris = cv2. cvtColor(image, cv2. COLOR_BGR2GRAY)
    gris = cv2. bitwise_not(gris)
    
    thresh = cv2.threshold(gris,  0,  255, cv2. THRESH_BINARY | cv2. THRESH_OTSU)[1]
    mser = cv2.MSER_create()
    coordinates, bboxes = mser.detectRegions(thresh)
    max_width1 = sorted(bboxes, key = lambda bboxes: (bboxes[0] + bboxes[2],bboxes[1] + bboxes[3]),
                     reverse = True)[0]
    
    max_width = list(max_width1)
    x,y,w,h = [i for i in max_width]
    roi = image[y:y+h,x:x+w]
    
    result = ocr.ocr(roi, det=False,rec = True, cls=False)

    long = len(result)
    
    for k in range(long):
        print(result[k])
        textos.append(str(result[k][0]))
    desc = ' '.join(textos)
    
    return desc