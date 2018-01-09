import numpy as np
import cv2
from code import *

img = generateImage(42, 4,3,5)
showAndWeight(img/255, '42')


for ii in range(0,128):
    img = generateImage(ii, 5,4,5)
    cv2.imwrite('img/'+str(ii)+'.png',img)
# img = generateImage(42, 4,3,5)
# cv2.imwrite('42.png',img)
# cv2.imshow('42',img)
# img = generateImage(49, 4,3,5)
# cv2.imwrite('49.png',img)
# cv2.imshow('49',img)
# while(True):
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

