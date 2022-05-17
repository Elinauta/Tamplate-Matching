import cv2
from cv2 import imread
import numpy as np

#Imagen a editar
imagen= cv2.imread(r'C:\Users\elycu\OneDrive\Escritorio\VisualStudio_Ejemplo\Practicas_Vision_7mo/Tuercas2.jpg',cv2.IMREAD_COLOR)

#GENERACION DE UNA NUEVA IMAGEN CON LA SELECCION ROI(REGION DE INTERES) DE OTRA IMAGEN
roi = cv2.selectROI(imagen)
img_recorte = imagen[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 

plantilla = img_recorte
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
plantilla_gris = cv2.cvtColor(plantilla, cv2.COLOR_BGR2GRAY)

w, h = plantilla_gris.shape[::-1]
res = cv2.matchTemplate(imagen_gris, plantilla_gris,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

#res = cv2.matchTemplate(imagen_gris, plantilla_gris, cv2.TM_SQDIFF)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#print(min_val, max_val, min_loc, max_loc)
#x1, y1 = min_loc
#x2, y2 = min_loc[0] + plantilla.shape[1], min_loc[1] + plantilla.shape[0]
#cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 3)

for pt in zip(*loc[::-1]):
    cv2.rectangle(imagen, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow("Image", imagen)
cv2.imshow("Template", plantilla)

cv2.waitKey(0)
cv2.destroyAllWindows()